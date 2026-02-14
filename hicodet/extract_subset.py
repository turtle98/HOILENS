import json
import sys
sys.path.append('/home/taehoon/HOILENS')
from utils.hico_text_label import hico_text_label

def box_area(box):
    """Calculate area of box [x1, y1, x2, y2]"""
    return (box[2] - box[0]) * (box[3] - box[1])

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box_area(box1)
    area2 = box_area(box2)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def get_unique_boxes(boxes, iou_threshold=0.9):
    """Deduplicate boxes using IoU threshold"""
    if not boxes:
        return []

    unique = [boxes[0]]
    for box in boxes[1:]:
        is_duplicate = False
        for u in unique:
            if compute_iou(box, u) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(box)

    return unique

def get_image_stats(annotation, size):
    """Get stats for an image: unique boxes, area ratios, etc."""
    img_width, img_height = size
    img_area = img_width * img_height

    if not annotation['boxes_h'] or not annotation['boxes_o']:
        return None

    unique_h = get_unique_boxes(annotation['boxes_h'])
    unique_o = get_unique_boxes(annotation['boxes_o'])
    total_unique = len(unique_h) + len(unique_o)

    max_ratio = 0
    for box in unique_h + unique_o:
        ratio = box_area(box) / img_area
        max_ratio = max(max_ratio, ratio)

    return {
        'unique_objects': len(unique_o),
        'total_annotations': total_unique,
        'max_area_ratio': max_ratio,
        'hoi_set': set(annotation['hoi'])
    }


def image_meets_tier(stats, max_area_ratio, max_objects, max_total_annotations):
    """Check if image stats meet the given tier criteria."""
    return (stats['unique_objects'] <= max_objects and
            stats['total_annotations'] < max_total_annotations and
            stats['max_area_ratio'] <= max_area_ratio)


def filter_subset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Build set of valid HOI IDs from hico_text_label
    # hico_text_label keys are (verb_idx, obj_idx), correspondence is (obj_idx, verb_idx)
    valid_hoi_ids = set()
    for _, (hoi_id, obj_idx, verb_idx) in enumerate(data['correspondence']):
        if (verb_idx, obj_idx) in hico_text_label:
            valid_hoi_ids.add(hoi_id)

    print(f"Valid HOI categories from hico_text_label: {len(valid_hoi_ids)}")

    # Define progressive relaxation tiers (strictest first)
    # Each tier: (max_area_ratio, max_objects, max_total_annotations)
    relaxation_tiers = [
        (0.20, 1, 4),   # Strictest: small boxes, 1 object, few annotations
        (0.30, 1, 4),   # Relax area
        (0.40, 1, 5),   # More area relaxation
        (0.20, 2, 5),   # Allow 2 objects
        (0.30, 2, 5),   # 2 objects + more area
        (0.40, 2, 6),   # Even more relaxed
        (0.50, 2, 6),   # Half image area OK
        (0.50, 3, 7),   # 3 objects allowed
        (0.60, 3, 8),   # More relaxed
        (0.70, 4, 10),  # Almost no restrictions
        (1.00, 30, 50), # Fallback: accept anything
    ]

    # Pre-compute stats for all images and build HOI -> images mapping
    print("Computing image statistics...")
    image_stats = {}  # idx -> stats
    hoi_to_images = {}  # hoi_id -> list of (idx, stats)

    for idx, (annotation, size) in enumerate(zip(data['annotation'], data['size'])):
        stats = get_image_stats(annotation, size)
        if stats:
            image_stats[idx] = stats
            for hoi_id in stats['hoi_set']:
                if hoi_id in valid_hoi_ids:  # Only track valid HOIs
                    if hoi_id not in hoi_to_images:
                        hoi_to_images[hoi_id] = []
                    hoi_to_images[hoi_id].append((idx, stats))

    print(f"Total valid images: {len(image_stats)}")
    print(f"HOI categories with images: {len(hoi_to_images)}")

    # For each valid HOI category, find the simplest image (lowest tier that has a match)
    selected_set = set()
    hoi_to_selected = {}  # hoi_id -> (idx, tier_idx)
    tier_counts = [0] * len(relaxation_tiers)

    for hoi_id in sorted(valid_hoi_ids):
        if hoi_id not in hoi_to_images:
            obj_idx, verb_idx = data['correspondence'][hoi_id]
            obj_name = data['objects'][obj_idx]
            verb_name = data['verbs'][verb_idx]
            print(f"WARNING: HOI {hoi_id} ({verb_name} {obj_name}) has no images in dataset!")
            continue

        candidates = hoi_to_images[hoi_id]
        selected_idx = None
        selected_tier = None

        # Try each tier from strictest to most relaxed
        for tier_idx, (max_area_ratio, max_objects, max_total_annotations) in enumerate(relaxation_tiers):
            # Find best candidate at this tier (prefer already-selected images for efficiency)
            best_idx = None
            best_is_already_selected = False

            for idx, stats in candidates:
                if image_meets_tier(stats, max_area_ratio, max_objects, max_total_annotations):
                    is_already_selected = idx in selected_set
                    # Prefer already-selected images to minimize total image count
                    if best_idx is None or (is_already_selected and not best_is_already_selected):
                        best_idx = idx
                        best_is_already_selected = is_already_selected
                    if best_is_already_selected:
                        break  # Can't do better than an already-selected image

            if best_idx is not None:
                selected_idx = best_idx
                selected_tier = tier_idx
                break

        if selected_idx is not None:
            selected_set.add(selected_idx)
            hoi_to_selected[hoi_id] = (selected_idx, selected_tier)
            tier_counts[selected_tier] += 1
        else:
            print(f"WARNING: Could not find any image for HOI {hoi_id}!")

    # Print tier statistics
    print(f"\n{'='*60}")
    print("HOI categories per tier:")
    for tier_idx, (max_area_ratio, max_objects, max_total_annotations) in enumerate(relaxation_tiers):
        if tier_counts[tier_idx] > 0:
            print(f"  Tier {tier_idx + 1} (area≤{max_area_ratio:.0%}, obj≤{max_objects}, ann<{max_total_annotations}): "
                  f"{tier_counts[tier_idx]} HOIs")

    selected_indices = sorted(selected_set)

    print(f"\nCoverage: {len(hoi_to_selected)}/{len(valid_hoi_ids)} valid HOI categories")
    if len(hoi_to_selected) < len(valid_hoi_ids):
        missing = valid_hoi_ids - set(hoi_to_selected.keys())
        print(f"Missing HOI categories: {sorted(missing)}")
        for hoi_id in sorted(missing):
            obj_idx, verb_idx = data['correspondence'][hoi_id]
            obj_name = data['objects'][obj_idx]
            verb_name = data['verbs'][verb_idx]
            print(f"  HOI {hoi_id}: {verb_name} {obj_name}")

    # Limit to N images
    selected_indices = selected_indices[:10]

    # Build filtered dataset
    filtered_data = {
        'annotation': [data['annotation'][i] for i in selected_indices],
        'filenames': [data['filenames'][i] for i in selected_indices],
        'size': [data['size'][i] for i in selected_indices],
        'empty': [],
        'objects': data['objects'],
        'verbs': data['verbs'],
        'correspondence': data['correspondence'],
    }

    print(f"\nOriginal images: {len(data['annotation'])}")
    print(f"Selected images: {len(selected_indices)}")

    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    filter_subset(
        '/home/taehoon/HOILENS/hicodet/instances_train2015.json',
        '/home/taehoon/HOILENS/hicodet/instances_train2015_subset_10.json'
    )
