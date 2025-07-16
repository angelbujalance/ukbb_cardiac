import torch
import torch.nn.functional as F
import nibabel as nib
import argparse
import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Crop Images', add_help=False)

    # Main arguments
    parser.add_argument('--data_path', default='/scratch-shared/abujalancegome/CMR_data', type=str,
                        help='Path to the CMR dicom files')
    parser.add_argument('--batch_size', default=50, type=int,
                        help='Number of images to process in a batch')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save intermediate results (defaults to data_path)')
    parser.add_argument('--half_precision', default=True, action='store_true',
                        help='Use half precision (float16) to reduce memory usage')
    parser.add_argument('--max_height', default=None)
    parser.add_argument('--max_width', default=None)

    return parser


def get_bounding_box(image):
    """Finds the bounding box of the nonzero region in a 2D image."""
    nonzero_indices = torch.nonzero(image.squeeze())  # Remove singleton dim if present
    if nonzero_indices.shape[0] == 0:  # No white pixels found
        return None  
    min_y, min_x = nonzero_indices.min(dim=0)[0]
    max_y, max_x = nonzero_indices.max(dim=0)[0]
    return min_y.item(), max_y.item(), min_x.item(), max_x.item()


def crop_image(seg_tensor, orig_img):
    """
    Finds the max bounding box over all time slices and crops all slices accordingly.
    tensor: (W, H, C, T) -> A 4D tensor where T is time, H is height, C is channels, and W is width.
    """
    W, H, C, T = seg_tensor.shape  # Extract dimensions

    # Initialize max bounding box
    min_y, min_x = H, W
    max_y, max_x = 0, 0

    # Iterate over time slices to find the global bounding box
    T = 1 # we are cropping just for the first time step
    for t in range(T):
        bbox = get_bounding_box(seg_tensor[:, :, 0, t])  # Squeeze out the channel dim
        if bbox:
            y1, y2, x1, x2 = bbox
            min_y, min_x = min(min_y, y1), min(min_x, x1)
            max_y, max_x = max(max_y, y2), max(max_x, x2)

    # Ensure the bounding box is valid
    if min_y >= max_y or min_x >= max_x:
        print("No white region detected in any slice.")
        return orig_img  # Return original tensor if no white region is found

    # Add padding to bounding box (with boundary checks)
    min_y = max(0, min_y-10)
    max_y = min(H, max_y+10)
    min_x = max(0, min_x-10)
    max_x = min(W, max_x+10)

    # Crop all slices using the max bounding box
    cropped_tensor = orig_img[min_y:max_y, min_x:max_x, :, :]
    return cropped_tensor


def process_single_image(idx, data_path, max_dims=None, half_precision=True):
    """Process a single image and return its tensor and dimensions"""
    try:
        # Load segmentation file
        seg_file_path = os.path.join(data_path, str(idx), "seg_sa.nii.gz")
        seg_img = nib.load(seg_file_path)
        seg_data = seg_img.get_fdata()
        _, _, c, t = seg_data.shape

        mid_slice = c // 2
        seg_tensor = torch.from_numpy(seg_data[:, :, mid_slice - 3 : mid_slice + 3, 0]).unsqueeze(2)

        # Load original image file
        img_file_path = os.path.join(data_path, str(idx), "sa.nii.gz")
        img = nib.load(img_file_path)
        orig_img = img.get_fdata()
        _, _, c, _ = orig_img.shape

        # 3D data
        orig_tensor = torch.from_numpy(orig_img[:, :, mid_slice - 3 : mid_slice + 3, t // 2]).unsqueeze(2)

        # Free memory
        del seg_data, orig_img

        # Crop image
        cropped_tensor = crop_image(seg_tensor, orig_tensor)

        # Free memory
        del seg_tensor, orig_tensor

        # Get the middle slice
        # slice_index = 0 # cropped_tensor.shape[3] // 2
        # TODO changed slcie_index to be ts=0 to all, check if the code works well
        temp_tensor = cropped_tensor[:, :, :, :].squeeze(2)

        # Free memory
        del cropped_tensor

        # Normalize the tensor
        min_val = torch.min(temp_tensor)
        max_val = torch.max(temp_tensor)
        norm_tensor = (temp_tensor - min_val) / (max_val - min_val + 1e-8)

        # Convert to half precision if requested
        if half_precision:
            norm_tensor = norm_tensor.half()

        # Return the tensor and its dimensions
        # print("norm_tensor shape:", norm_tensor.shape)
        return norm_tensor, norm_tensor.shape

    except Exception as e:
        print(f"Error processing {idx}: {e}")
        return None, None


def save_batch(batch_tensors, batch_idx, output_path):
    """Save a batch of processed tensors"""
    if not batch_tensors:
        return
    
    batch_tensor = torch.stack(batch_tensors)
    batch_filename = os.path.join(output_path, f'cmr_batch_systole_{batch_idx}.pt')
    torch.save(batch_tensor, batch_filename)
    print(f"Saved batch {batch_idx} with {len(batch_tensors)} images to {batch_filename}")
    
    # Free memory
    del batch_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_max_dimensions(data_list, data_path, max_height, max_width,
                        sample_size=1000):
    """Sample a subset of images to determine max dimensions"""
    print("Sampling images to determine max dimensions...")
    if max_height is not None and max_width is not None:
        max_height = int(max_height)
        max_width = int(max_width)
    else:
        max_height, max_width = 0, 0

        # Take a sample of the data list to determine dimensions
        sample_indices = np.random.choice(len(data_list), 
                                        min(sample_size, len(data_list)), 
                                        replace=False)

        for i in tqdm(sample_indices):
            idx = data_list[i]
            _, dims = process_single_image(idx, data_path)
            if dims is not None:
                max_height = max(max_height, dims[0])
                max_width = max(max_width, dims[1])

        # Add a small buffer (10%) to accommodate potential larger images
        max_height = int(max_height * 1.1)
        max_width = int(max_width * 1.1)

    print(f"Estimated max dimensions: {max_height} x {max_width}")
    return max_height, max_width


def main(args):
    # Set up output directory
    output_dir = args.output_dir if args.output_dir else args.data_path
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all data directories
    cmr_pretrain_ids = list(pd.read_csv('/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/labels/ids.csv',
                                    header=None)[0])
    cmr_pretrain_ids = [str(id_) for id_ in cmr_pretrain_ids]

    data_list = [entry.name for entry in os.scandir(args.data_path)
                 if entry.is_dir() and entry.name in cmr_pretrain_ids]
    data_list.sort()

    del cmr_pretrain_ids

    print(f"Found {len(data_list)} potential image directories")

    # pd.DataFrame(empty_files).to_csv("empty_files_idx.csv", index=False, header=False)

    # Sample images to determine max dimensions
    max_height, max_width = find_max_dimensions(data_list, args.data_path,
                                                args.max_height, args.max_width)

    # Process images in batches
    batch_tensors = []
    batch_idx = 0
    total_processed = 0

    for idx in tqdm(data_list, desc="Processing images"):
        tensor, _ = process_single_image(idx, args.data_path, half_precision=args.half_precision)

        if tensor is not None:
            if tensor.shape[2] < 6:
                print(f"Not enough dimensions for index {idx}")
                continue
            # Pad the tensor to the maximum size
            pad_height = max_height - tensor.size(0)
            pad_width = max_width - tensor.size(1)

            # Calculate padding for each side
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # Permute to [T, H, W]
            tensor = tensor.permute(2, 0, 1)

            # Pad the tensor
            # print("tensor shape:", tensor.shape)
            padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
            # print("padded_tensor shape:", padded_tensor.shape)

            batch_tensors.append(padded_tensor)

            # Free memory
            del tensor

            total_processed += 1

            # Save batch when it reaches the specified size
            if len(batch_tensors) >= args.batch_size:
                save_batch(batch_tensors, batch_idx, output_dir)
                batch_tensors = []
                batch_idx += 1

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save any remaining tensors in the last batch
    if batch_tensors:
        print(batch_tensors)
        save_batch(batch_tensors, batch_idx, output_dir)

    # Create metadata file with information about the batches
    metadata = {
        'total_images': total_processed,
        'batches': batch_idx + 1,
        'batch_size': args.batch_size,
        'max_height': max_height,
        'max_width': max_width,
        'half_precision': args.half_precision
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, 'cmr_processing_metadata.pt')
    torch.save(metadata, metadata_path)

    print(f"Processing complete. Processed {total_processed} images across {batch_idx + 1} batches.")
    print(f"Maximum dimensions: {max_height} x {max_width}")
    print(f"Metadata saved to {metadata_path}")


def combine_batches(output_dir):
    """
    Utility function to combine all batches into a single tensor file if needed
    Only call this if your system has sufficient memory
    """
    batch_files = [f for f in os.listdir(output_dir) if f.startswith('cmr_batch_systole_') and f.endswith('.pt')]
    batch_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))

    print("trying to combine all batches together...")

    all_tensors = []
    for batch_file in tqdm(batch_files, desc="Combining batches"):
        batch_tensor = torch.load(os.path.join(output_dir, batch_file))
        all_tensors.append(batch_tensor)
        del batch_tensor  # Free memory immediately
        gc.collect()

    # Concatenate all batches
    combined_tensor = torch.cat(all_tensors, dim=0)

    # Save combined tensor
    torch.save(combined_tensor, os.path.join(output_dir, 'cmr_tensors_ts_all_systole.pt'))
    print(f"Combined tensor saved with shape: {combined_tensor.shape}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)

    combine_batches(args.data_path)