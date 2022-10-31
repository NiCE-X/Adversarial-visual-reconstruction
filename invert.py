# python 3.6

import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image

from data_loader import get_loader
from models.stylegan_generator_network import StyleGANGeneratorNet

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='styleganinv_ffhq256', help='Name of the GAN model.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--celeba_image_dir', type=str, default="/hd6/guanweinan/Data/CelebAMask_HQ/CelebAMask-HQ/CelebA-HQ-img/",
                      help='path for celeb data')
  parser.add_argument('--attr_path', type=str, default="/hd6/guanweinan/Data/CelebAMask_HQ/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt",
                      help='path for celeb anno')
  parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
  parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  output_dir = args.output_dir or f'results/inversion'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      logger=logger)
  image_size = inverter.G.resolution
  
  #load celeba
  data_loader = get_loader(args.celeba_image_dir, args.attr_path, args.selected_attrs, num_workers=4)
  for img_idx, (x_real, c_org, filename) in enumerate(data_loader):
    image = (255 * x_real[0].numpy().transpose(1, 2, 0)).astype(np.uint8)
    
    hair_color_indices = []
    for i, attr_name in enumerate(args.selected_attrs):
      if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
        hair_color_indices.append(i)
    label = []
    for i in range(args.c_dim):
      c_trg = c_org.clone()
      if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
          c_trg[:, i] = 1
          for j in hair_color_indices:
              if j != i:
                  c_trg[:, j] = 0
      else:
          c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.    
      label.append(c_trg.cuda())
    
    code, viz_results, stargan_results = inverter.easy_invert(image, label, num_viz=args.num_results)
    image_name = os.path.splitext(os.path.basename(filename[0]))[0]
    
    save_image(f'{output_dir}/{image_name}_ori.png', viz_results[0])
    save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
    save_image(f'{output_dir}/{image_name}_inv.png', viz_results[-1])
    os.makedirs(f'{output_dir}/stargan', exist_ok=True)
    for num in range(len(args.selected_attrs)):
      save_image(f'{output_dir}/stargan/{image_name}_rec_{num}.png', stargan_results[num])
      save_image(f'{output_dir}/stargan/{image_name}_ori_{num}.png', stargan_results[num+len(args.selected_attrs)])

if __name__ == '__main__':
  main()
