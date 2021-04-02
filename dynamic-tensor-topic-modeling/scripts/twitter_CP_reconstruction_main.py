from covid19.online_CPDL.tweets_reconstruction_OCPDL import Tweets_Reconstructor_OCPDL
import numpy as np
from itertools import product
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # sources = ['Data/Twitter/top_1000_daily/data_tensortweets_from_2020-02-01-00_to_2020-05-01-00-003.pickle']
    # sources = ['Data/Twitter/random_1000_daily/data_tensor1000_random_tweets_daily_from_2020-02-01-00_to_2020-05-01-00-002.pickle']
    sources = ['Data/Twitter/top_1000_daily/data_tensor_top1000.pickle',
               'Data/Twitter/random_1000_daily/data_tensor_random1000.pickle']

    # sources = ['Data/Twitter/top_100_weekly/tweets_data_tensor.pickle']
    # save_file_name = 'weekly100'
    # save_file_name = 'random1000_sparsity2_20topics'

    # for path in sources:
    for i in np.arange(2):
        path = sources[i]

        n_topics = 10
        n_iter = 30
        sparsity = 2
        batch_size = 30
        segment_length = batch_size
        seq_refresh_history_as = 2

        if i == 0:
            save_file_name = 'top1000' + '_ntopics_' + str(n_topics) + '_iter_' + str(n_iter) + '_sparsity_' + str(
                sparsity) + '_batchsize_' + str(batch_size) + '_ref_history_as_' + str(seq_refresh_history_as)
        else:
            save_file_name = 'random1000' + '_ntopics_' + str(n_topics) + '_iter_' + str(n_iter) + '_sparsity_' + str(
                sparsity) + '_batchsize_' + str(batch_size) + '_ref_history_as_' + str(seq_refresh_history_as)

        reconstructor = Tweets_Reconstructor_OCPDL(path=path,
                                                   n_components=n_topics,  # number of dictionary elements -- rank
                                                   iterations=n_iter,  # number of iterations for the ONTF algorithm
                                                   sub_iterations=2,
                                                   # number of i.i.d. subsampling for each iteration of ONTF
                                                   batch_size=1,  # number of patches used in i.i.d. subsampling
                                                   num_patches=1,
                                                   # number of patches that the algorithm learns from at each iteration
                                                   segment_length=batch_size,
                                                   alpha=sparsity,
                                                   unfold_words_tweets=False)

        train_fresh = False
        display_dictionary = False
        train_sequential_fresh = True  ### need to set if_sample_from_tweets_mode = False
        display_sequential_dictionary = True
        if_reconstruct = False

        if train_fresh:
            W, At, Bt, H = reconstructor.train_dict(save_file_name=save_file_name,
                                                    if_sample_from_tweets_mode=True,
                                                    beta=1,
                                                    ini_history=1,
                                                    if_save=True)
            CPdict = reconstructor.out(W)
            print('W', W)
            print('W.keys()', W.keys())
            print('CPdict.keys()', CPdict.keys())
            print('U0.shape', W.get('U0').shape)
            print('U1.shape', W.get('U1').shape)
        # else:
        #    path = 'Tweets_dictionary/dict_learned_CPDL_' + save_file_name + '.npy'
        #    W = np.load(path, allow_pickle=True).item()

        if display_dictionary:
            path = 'Tweets_dictionary/dict_learned_CPDL_' + save_file_name + '.npy'
            W = np.load(path, allow_pickle=True).item()

            ### Wordclod representation
            reconstructor.display_dictionary_CP(W,
                                                save_fig_name=save_file_name,
                                                num_top_words_from_each_topic=20,
                                                num_word_sampling_from_topics=10000,
                                                if_plot=True)

            ### Grayscale representation
            topics_latex = reconstructor.display_topics_bar(W)
            print(topics_latex)

        if train_sequential_fresh:
            seq_dict = reconstructor.train_sequential_dict(save_file_name=save_file_name,
                                                           slide_window_by=segment_length,
                                                           refresh_history_as=seq_refresh_history_as,
                                                           beta=1)
            print('seq_dict.keys()', seq_dict.keys())
        # else:
        # path = 'Tweets_dictionary/sequential_dict_learned_CPDL_' + save_file_name + '.npy'
        # seq_dict = np.load(path, allow_pickle=True).item()

        if display_sequential_dictionary:
            path = 'Tweets_dictionary/sequential_dict_learned_CPDL_' + save_file_name + '.npy'
            seq_dict = np.load(path, allow_pickle=True).item()

            reconstructor.display_sequential_dictionary_CP(seq_dict=seq_dict,
                                                           save_fig_name=save_file_name,
                                                           num_top_words_from_each_topic=5,
                                                           num_word_sampling_from_topics=10000,
                                                           starting_topic_idx=0,
                                                           ending_topic_idx=None,
                                                           if_plot=True,
                                                           if_sample_topic_words=False)

        if if_reconstruct:
            path = 'Image_dictionary/dict_learned_CPDL_klimpt_allfold.npy'
            loading = np.load(path, allow_pickle=True).item()
            IMG_recons = reconstructor.reconstruct_image_color(loading=loading, recons_resolution=5, if_save=False)


if __name__ == '__main__':
    main()