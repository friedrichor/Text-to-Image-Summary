import os
import sys
import json
import argparse

import datasets
from huggingface_hub import notebook_login, create_repo, Repository


def get_paramters():
    parser = argparse.ArgumentParser(description='Upload dataset')
    parser.add_argument('--save_to_disk', 
                        default=True,
                        help='Save to disk instead of in HuggingFace')
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='Text2Image_example',
                        help='name of the dataset')
    parser.add_argument('--user_name', 
                        type=str,
                        help='user name in huggingface')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_paramters()
    images_folder = os.path.join(sys.path[0], "images")
    captions_file = os.path.join(sys.path[0], "image_captions.json")
    
    with open(captions_file, "r") as f:
        data_dic = json.load(f)
    
    dataset_dict = {
        'image': list(data_dic.keys()),
        'text': list(data_dic.values())
    }
    features = datasets.Features({
        'image': datasets.Image(),
        'text': datasets.Value('string'),
    })
    
    # Create dataset
    dataset = datasets.Dataset.from_dict(dataset_dict, features, split="train")
    print(dataset)  # 测试输出
    print(dataset[0])  # 测试输出

    # Save dataset
    # Localy
    if args.save_to_disk:
        dataset.save_to_disk(os.path.join(sys.path[0], f"datasets/{args.dataset_name}"))
    # Push dataset to HuggingFace 
    else:
        repo_url = create_repo(repo_id=args.dataset_name, repo_type="dataset", exist_ok=True)
        dataset.push_to_hub(f'{args.user_name}/{args.dataset_name}')
