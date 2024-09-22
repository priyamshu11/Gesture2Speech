import json
import random
import os

def convert_keypoints_to_floats(keypoints_data):
    for label, keypoints_list in keypoints_data.items():
        for i, frame_keypoints in enumerate(keypoints_list):
            keypoints_data[label][i] = [[float(coord) for coord in kp] for kp in frame_keypoints]
    return keypoints_data

def augment_keypoints(keypoints_data, num_augmented_versions=5, frame_keep_prob=0.7, noise_factor=0.05):
    augmented_data = {}
    for label, keypoints_list in keypoints_data.items():
        augmented_data[label] = []
        for _ in range(num_augmented_versions):
            augmented_version = []
            for frame_keypoints in keypoints_list:
                if random.random() < frame_keep_prob:
                    noisy_frame = [[kp[0] + random.uniform(-noise_factor, noise_factor),  # X-axis
                                   kp[1] + random.uniform(-noise_factor, noise_factor),  # Y-axis
                                   kp[2] + random.uniform(-noise_factor, noise_factor)]  # Z-axis
                                  for kp in frame_keypoints]
                    augmented_version.append(noisy_frame)
            if len(augmented_version) == 0:
                augmented_version = keypoints_list[:]
            augmented_data[label].append(augmented_version)
    return augmented_data

def load_keypoints_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def save_augmented_data(augmented_data, base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    for label, augmented_versions in augmented_data.items():
        label_folder = os.path.join(base_output_dir, label)
        os.makedirs(label_folder, exist_ok=True)
        for version_idx, augmented_version in enumerate(augmented_versions):
            version_folder = os.path.join(label_folder, f'version_{version_idx + 1}')
            os.makedirs(version_folder, exist_ok=True)
            filename = 'keypoints.json'
            output_path = os.path.join(version_folder, filename)
            with open(output_path, 'w') as outfile:
                json.dump({label: augmented_version}, outfile)

def main():
    input_file = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\normalized_keypoints.json'  # Adjust this path
    keypoints_data = load_keypoints_json(input_file)

    # Convert keypoints to floats
    keypoints_data = convert_keypoints_to_floats(keypoints_data)

    # Augment the keypoints data
    augmented_data = augment_keypoints(keypoints_data, num_augmented_versions=20, frame_keep_prob=0.7, noise_factor=0.05)

    # Save augmented data to a structured folder
    save_augmented_data(augmented_data, base_output_dir="augmented_data")

if __name__ == "__main__":
    main()