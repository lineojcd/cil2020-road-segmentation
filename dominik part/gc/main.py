import argparse
import os
from graph_cut import GraphCut
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.sparse import coo_matrix
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lambda_val', action='append', default=[0], type=float)
    parser.add_argument('-ip', '--in_dir_pred', required=True)
    parser.add_argument('-io', '--in_dir_orig', default='data/test_images')
    parser.add_argument('-v', '--verbose', default='True')
    args = parser.parse_args()

    if (len(args.lambda_val) > 1):
        lambdas = args.lambda_val[1:]
    else:
        lambdas = args.lambda_val

    # Assumption: Input directories contain only images, corresponding images have the same name
    if not os.path.exists(args.in_dir_pred) or not os.path.exists(args.in_dir_orig):
        print('Invalid input directory!')
        exit()
    img_names = [name for name in os.listdir(args.in_dir_pred) if os.path.isfile(os.path.join(args.in_dir_pred, name))]

    if args.in_dir_pred[-1] == '/':
        out_dir = args.in_dir_pred[:-1] + '_gc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    wb = 20                                       # width of barrier between images
    width = 608                                   # width of images

    for lambda_value in tqdm(lambdas):

        if not os.path.exists(os.path.join(out_dir, str(lambda_value))):
            out_base_path = os.path.join(out_dir, str(lambda_value))
            os.makedirs(out_base_path)

        for name in tqdm(img_names):

            img_orig = Image.open(os.path.join(args.in_dir_orig, name))
            img_pred = Image.open(os.path.join(args.in_dir_pred, name))
            img_gc = segment_image(img_orig, img_pred, lambda_value)

            if args.verbose in ['True', 'true', 't', 'T', '1', 'yes', 'y', 'Yes']:
                Imgs = np.zeros((width, 3 * width + 2 * wb, 3), dtype=np.int)
                Imgs[:, :width, :] = img_orig
                Imgs[:, width : width+wb, :2] = 0
                Imgs[:, width : width+wb, 2] = 255
                Imgs[:, width+wb : 2*width+wb, 0] = img_pred
                Imgs[:, 2*width+wb : 2*(width+wb), :2] = 0
                Imgs[:, 2*width+wb : 2*(width+wb), 2] = 255
                Imgs[:, 2*(width+wb):, 0] = img_gc

                plt.imshow(Imgs)
                plt.title('original (left), nn-prediction (middle), graphcut (right)')
                plt.show()

            path = os.path.join(out_base_path, name)
            Image.fromarray(img_gc).save(path)


def segment_image(img_orig, img_pred, lambda_value):
    image_array = np.asarray(img_orig)
    pred_mask = np.asarray(img_pred)

    height, width = np.shape(image_array)[0:2]
    num_pixels = height * width

    # compute percentage of points we need to sample to get approximately the same amount of negative points as positive points
    # if we take all positive points for sure
    percent_pos = np.sum(pred_mask > 0) / num_pixels
    percent_needed = percent_pos * (1 / (1 - percent_pos))

    if percent_pos > 0.5:
        print('In the input prediction more than 50% of the points are marked as street pixel, you need to change the seeding to handle such cases.')
        print('Keeping initial prediction!')
        return np.copy(img_pred)

    # create random numbery by twice percentage
    r = np.random.rand(height, width)
    subset_mask = r < percent_needed

    rows, cols = np.where(pred_mask > 0)
    seed_fg = np.stack((rows, cols)).T

    rows, cols = np.where((pred_mask == 0) * subset_mask)
    seed_bg = np.stack((rows, cols)).T

    # Get the color histogram for the unaries
    # print("Calculating color histograms...")
    hist_res = 32
    cost_fg = __get_color_histogram(image_array, seed_fg, hist_res)
    cost_bg = __get_color_histogram(image_array, seed_bg, hist_res)

    # Set the unaries and the pairwise terms
    # print("Calculating terminal weights...")
    unaries = __get_unaries(
        image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
    # print("Calculating non-terminal weights...")
    adj_mat = __get_pairwise(image_array)

    # Done: TASK 2.4 - perform graph cut
    # Your code here
    gc = GraphCut(num_pixels, adj_mat.count_nonzero() * 2)
    # print("Setting terminal weights...")
    gc.set_unary(unaries)
    # print("Setting non-terminal weights...")
    gc.set_neighbors(adj_mat)
    # print("Calculate min-cut...")
    gc.minimize()
    labels = gc.get_labeling()

    labels = np.reshape(labels, (height, width))
    # plt.imshow(labels)
    # plt.show()

    img_gc = labels * 255

    return img_gc.astype(np.uint8)


def __get_color_histogram(image, seed, hist_res):

    chosen_ps = image[seed[:,0], seed[:,1]]  # check if i have to switch it

    histogram = np.zeros((hist_res, hist_res, hist_res))
    indexes_rgb = (chosen_ps / 256 * hist_res).astype(int)
    for r, g, b in indexes_rgb:
        histogram[r][g][b] += 1

    # if sigma is in [0.625, 0.875) the default of truncate = 4 makes sure that the window size is 7
    histogram_s = ndimage.gaussian_filter(histogram, 0.7)
    histogram_sn = histogram_s / np.sum(histogram_s)
    # histogram_sn[histogram_sn < 1e-10] = 1e-10
    histogram_sn += 1e-10

    return histogram_sn

def __get_unaries(image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):

    h = image.shape[0]
    w = image.shape[1]
    unaries = np.empty((h, w, 2))

    hist_res = hist_fg.shape[0]
    indexes_rgb = (image / 256 * hist_res).astype(int)
    for row in range(h):
        for col in range(w):
            r, g, b = indexes_rgb[row, col]
            cost_fg = lambda_param * -np.log(hist_fg[r,g,b])
            cost_bg = lambda_param * -np.log(hist_bg[r,g,b])
            unaries[row, col] = [cost_bg, cost_fg]

    for row, col in seed_bg:
        unaries[row, col] = [0, np.inf]

    for row, col in seed_fg:
        unaries[row, col] = [np.inf, 0]

    return unaries.reshape((h*w, 2))


def __get_pairwise(image):
 
    sigma = 5
    two_sig_sq = 2 * sigma**2

    h, w = image.shape[:2]
    # n_entries = 4 * h * w - 3 * h - 3 * w + 2

    # compute all horizonal weights
    intensity_ssd = np.sum(np.square(image[:,:-1] - image[:,1:]), axis=2)
    # euclidean distance between pixels is always 1
    Bqp = np.exp(-(intensity_ssd / two_sig_sq))
    data_hor = Bqp.reshape(-1)
    cols_hor = np.arange(1,w*h+1).reshape(h,w)[:,:-1].reshape(-1)
    rows_hor = cols_hor - 1

    # compute all vertical weights
    intensity_ssd = np.sum(np.square(image[:-1,:] - image[1:,:]), axis=2)
    # euclidean distance between pixels is always 1
    # Bqp contains weights for each pixel with pixel on its right
    Bqp = np.exp(-(intensity_ssd / two_sig_sq))
    data_ver = Bqp.reshape(-1)
    rows_ver = np.arange(w*(h-1))
    cols_ver = rows_ver + w

    # compute all upper left to lower right diagonal weights (and vice versa)
    intensity_ssd = np.sum(np.square(image[:-1,:-1] - image[1:,1:]), axis=2)
    dist_scale = 1 / np.sqrt(2)
    Bqp = np.exp(-(intensity_ssd / two_sig_sq)) * dist_scale
    data_diag0 = Bqp.reshape(-1)
    rows_diag0 = rows_hor[:-w+1]
    cols_diag0 = rows_diag0 + w + 1

    # compute all lower left to upper right diagonal weights (and vice versa)
    intensity_ssd = np.sum(np.square(image[1:,:-1] - image[:-1,1:]), axis=2)
    # euclidean distance between pixels is again sqrt(2)
    Bqp = np.exp(-(intensity_ssd / two_sig_sq)) * dist_scale
    data_diag1 = Bqp.reshape(-1)
    rows_diag1 = rows_hor[:-w+1] + 1
    cols_diag1 = cols_diag0 - 1

    data = np.hstack((data_hor, data_ver, data_diag0, data_diag1))
    rows = np.hstack((rows_hor, rows_ver, rows_diag0, rows_diag1))
    cols = np.hstack((cols_hor, cols_ver, cols_diag0, cols_diag1))

    # assert(data.shape[0] == n_entries)
    # assert(rows.shape[0] == n_entries)
    # assert(cols.shape[0] == n_entries)

    # Only triu is filled since it is symmetric and only that part is used later on
    adj_mat = coo_matrix((data, (rows, cols)), shape=(h*w, h*w))

    return adj_mat

if __name__ == "__main__":
    main()
