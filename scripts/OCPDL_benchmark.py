from utils.ocpdl import Online_CPDL
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from headlines_preprocessing import generate_tensor
from utils.tweets_reconstruction_OCPDL import Tweets_Reconstructor_OCPDL
import os
from tensorly.decomposition import non_negative_parafac


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def initialize_loading(X, n_components):
    ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
    loading = {}
    n_modes = len(X.shape)
    for i in np.arange(n_modes):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
        loading.update({'U' + str(i): np.random.rand(X.shape[i], n_components)})
    return loading


def Out_tensor(loading):
    ### given loading, take outer product of respected columns to get CPdict
    CPdict = {}
    n_modes = len(loading.keys())
    n_components = loading.get('U0').shape[1]
    print('!!! n_modes', n_modes)
    print('!!! n_components', n_components)

    for i in np.arange(n_components):
        A = np.array([1])
        for j in np.arange(n_modes):
            loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
            # print('loading_factor', loading_factor)
            A = np.multiply.outer(A, loading_factor[:, i])
        A = A[0]
        CPdict.update({'A' + str(i): A})
    print('!!! CPdict.keys()', CPdict.keys())

    X = np.zeros(shape=CPdict.get('A0').shape)
    for j in np.arange(len(loading.keys())):
        X += CPdict.get('A' + str(j))

    return X

def MU_run(X,
           n_components=10,
           iter=100,
           regularizer=0,
           ini_loading=None,
           if_compute_recons_error=True,
           save_folder='Output_files',
           output_results=True):
    ALSDR = ALS_DR(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = ALSDR.MU(iter=iter,
                           ini_loading=ini_loading,
                           if_compute_recons_error=if_compute_recons_error,
                           save_folder=save_folder,
                           output_results=output_results)
    return result_dict

def OCPDL_run(X,
              n_components=10,
              iter=100,
              regularizer=0,
              ini_loading=None,
              batch_size=100,
              beta=None,
              mode_2be_subsampled=-1,
              if_compute_recons_error=True,
              save_folder='Output_files',
              output_results=True):
    OCPDL = Online_CPDL(X=X,
                        batch_size=batch_size,
                        iterations=iter,
                        n_components=n_components,
                        ini_loading=ini_loading,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer,
                        beta=beta,
                        subsample=True)

    result_dict = OCPDL.train_dict(mode_2be_subsampled=mode_2be_subsampled,
                                   if_compute_recons_error=if_compute_recons_error,
                                   save_folder=save_folder,
                                   output_results=output_results)
    return result_dict


def plot_benchmark_errors_list(ALS_results_list_dict=None, OCPDL_result=None, MU_result=None, name=1, save_folder=None):

    if ALS_results_list_dict is not None:
        ALS_results_names = [name for name in ALS_results_list_dict.keys()]
        d1 = ALS_results_list_dict.get(ALS_results_names[0])
        n_components = d1.get('n_components')
        print('!!! ALS_results_names', ALS_results_names)

        time_records = {}
        ALS_errors = {}
        f_ALS_interpolated = {}
        # for i in np.arange(len(ALS_results_names)):
        #      ALS_errors.update({'ALS_errors' + str(i) : results_list_dict.get(str(ALS_results_names[i])).get('timed_errors_trials')})

        # max duration
        x_all_max = 0
        for i in np.arange(len(ALS_results_names)):
            ALS_errors0 = ALS_results_list_dict.get(ALS_results_names[i]).get('timed_errors_trials')
            x_all_max = max(x_all_max, max(ALS_errors0[:, :, -1][:, 0]))

        x_all = np.linspace(0, x_all_max, num=101, endpoint=True)

        for i in np.arange(len(ALS_results_names)):
            ALS_errors0 = ALS_results_list_dict.get(ALS_results_names[i]).get('timed_errors_trials')
            time_records.update({'x_all_ALS'+ str(i) : x_all[x_all < min(ALS_errors0[:, :, -1][:, 0])]})


        # interpolate data and have common carrier
        for j in np.arange(len(ALS_results_names)):
            f_ALS_interpolated0 = []
            ALS_errors0 = ALS_results_list_dict.get(ALS_results_names[j]).get('timed_errors_trials')
            for i in np.arange(ALS_errors0.shape[0]):
                f_ALS0 = interp1d(ALS_errors0[i, 0, :], ALS_errors0[i, 1, :], fill_value="extrapolate")
                x_all_ALS0 = time_records.get('x_all_ALS'+ str(j))
                f_ALS_interpolated0.append(f_ALS0(x_all_ALS0))

            f_ALS_interpolated0 = np.asarray(f_ALS_interpolated0)
            f_ALS_interpolated.update({'f_ALS_interpolated'+str(j): f_ALS_interpolated0})

    # x_all_common = x_all_ALS1[range(np.round(len(x_all_ALS1) // 1.1).astype(int))]
    # x_all_MU = x_all_common
    if MU_result is not None:
        MU_errors = MU_result.get('timed_errors_trials')
        if ALS_results_list_dict is None:
            x_all_max = 0
            x_all_max = max(x_all_max, max(MU_errors[:, :, -1][:, 0]))
            x_all = np.linspace(0, x_all_max, num=101, endpoint=True)
            n_components = MU_result.get('n_components')

        x_all_MU = x_all[x_all < min(MU_errors[:, :, -1][:, 0])]
        n_trials = MU_errors.shape[0]

        f_MU_interpolated = []
        for i in np.arange(MU_errors.shape[0]):
            f_MU = interp1d(MU_errors[i, 0, :], MU_errors[i, 1, :], fill_value="extrapolate")
            f_MU_interpolated.append(f_MU(x_all_MU))
        f_MU_interpolated = np.asarray(f_MU_interpolated)
        f_MU_avg = np.sum(f_MU_interpolated, axis=0) / f_MU_interpolated.shape[0]  ### axis-0 : trials
        f_MU_std = np.std(f_MU_interpolated, axis=0)

    if OCPDL_result is not None:
        OCPDL_errors = OCPDL_result.get('timed_errors_trials')
        x_all_OCPDL = x_all[x_all < min(OCPDL_errors[:, :, -1][:, 0])]
        n_trials = OCPDL_errors.shape[0]

        f_OCPDL_interpolated = []
        for i in np.arange(OCPDL_errors.shape[0]):
            f_OCPDL = interp1d(OCPDL_errors[i, 0, :], OCPDL_errors[i, 1, :], fill_value="extrapolate")
            f_OCPDL_interpolated.append(f_OCPDL(x_all_OCPDL))
        f_OCPDL_interpolated = np.asarray(f_OCPDL_interpolated)
        f_OCPDL_avg = np.sum(f_OCPDL_interpolated, axis=0) / f_OCPDL_interpolated.shape[0]  ### axis-0 : trials
        f_OCPDL_std = np.std(f_OCPDL_interpolated, axis=0)

    # make figure
    color_list = ['r', 'c', 'k']
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    if ALS_results_list_dict is not None:
        for i in np.arange(len(ALS_results_names)):

            f_ALS_interpolated0 = f_ALS_interpolated.get('f_ALS_interpolated' + str(i))
            f_ALS_avg0 = np.sum(f_ALS_interpolated0, axis=0) / f_ALS_interpolated0.shape[0]  ### axis-0 : trials
            f_ALS_std0 = np.std(f_ALS_interpolated0, axis=0)
            # print('!!!! f_ALS_avg0_' + str(i), f_ALS_avg0)

            x_all_ALS0 = time_records.get('x_all_ALS'+ str(i))
            color = color_list[i % len(color_list)]
            markers, caps, bars = axs.errorbar(x_all_ALS0, f_ALS_avg0, yerr=f_ALS_std0,
                                               fmt=color+'-', marker='*', label=ALS_results_names[i], errorevery=5)
            axs.fill_between(x_all_ALS0, f_ALS_avg0 - f_ALS_std0, f_ALS_avg0 + f_ALS_std0, facecolor=color, alpha=0.1)

    if MU_result is not None:
        # Add MU plots
        markers, caps, bars = axs.errorbar(x_all_MU, f_MU_avg, yerr=f_MU_std,
                                           fmt='g-', marker='x', label='NCPD', errorevery=5)
        axs.fill_between(x_all_MU, f_MU_avg - f_MU_std, f_MU_avg + f_MU_std, facecolor='g', alpha=0.2)

    if OCPDL_result is not None:
        # Add OCPDL plots
        markers, caps, bars = axs.errorbar(x_all_OCPDL, f_OCPDL_avg, yerr=f_OCPDL_std,
                                           fmt='b-', marker='x', label='Online NCPD', errorevery=5)
        axs.fill_between(x_all_OCPDL, f_OCPDL_avg - f_OCPDL_std, f_OCPDL_avg + f_OCPDL_std, facecolor='b', alpha=0.2)


    if ALS_results_list_dict is None:
        # min_max duration
        x_all_min_max = []
        MU_times = x_all[x_all < min(MU_errors[:, :, -1][:, 0])]
        OCPDL_times = x_all[x_all < min(OCPDL_errors[:, :, -1][:, 0])]
        x_all_min_max.append(MU_times[-1])
        x_all_min_max.append(OCPDL_times[-1])

        x_all_min_max = min(x_all_min_max)
        axs.set_xlim(0, x_all_min_max)


    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=14)
    axs.set_ylabel('Reconstruction error', fontsize=12)
    plt.suptitle('Reconstruction error benchmarks')
    axs.legend(fontsize=13)
    plt.tight_layout()
    plt.suptitle('Reconstruction error benchmarks', fontsize=13)
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)
    if save_folder is None:
        root = 'Output_files_BCD'
    else:
        root = save_folder

    plt.savefig(root + '/benchmark_plot_errorbar' + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
        n_components) + "_" + str(name) + ".pdf")



def main():
    loading = {}
    n_components = 25
    iter = 400
    num_repeat = 1
    save_folder = "Output_files/test5"

    run_ALS = False
    run_MU = False
    run_OCPDL = True
    plot_errors = False
    plot_dictionary = False
    # search_radius_const = 100000
    file_identifier = '1'

    synthetic_data = False
    News_20 = False
    headlines = True
    COVID_twitter = False

    # Load data

    if synthetic_data:
        np.random.seed(1)
        U0 = np.random.rand(100, n_components)
        np.random.seed(2)
        U1 = np.random.rand(100, n_components)
        np.random.seed(3)
        U2 = np.random.rand(500, n_components)

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        loading.update({'U2': U2})
        file_name = "Synthetic"
        X = Out_tensor(loading)

    if COVID_twitter:
        path = "Data/Twitter/top_1000_daily/data_tensor_top1000.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict[1]
        file_name = "Twitter"

    if News_20:
        path = "Data/20news_tfidf_tensor.pickle"
        X = pickle.load(open(path, "rb"))
        file_name = "News20"

    if headlines:
        path = "Data/abcnews-date-text.csv"
        # X = generate_tensor(path)
        # path = "Data/headlines_tensor.pickle"
        # X = pickle.load(open(path, "rb"))
        dict = generate_tensor(path)
        X = dict[1]
        file_name = "Headlines"



    # initialize loading matrices
    loading_list = []
    for i in np.arange(num_repeat):
        W = initialize_loading(X, n_components)
        loading_list.append(W.copy())


    file_name = file_name + "_" + file_identifier
    print('X.shape', X.shape)
    print('file_name', str(file_name))
    normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape)
    data_scale_factor = 10000
    search_radius_const = 10*data_scale_factor*np.linalg.norm(X.reshape(-1,1),1)
    # print('!!! average entry size of tensor:', np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape))
    X = X * data_scale_factor

    if run_ALS:
        ALS_result_list_dict = {}
        ALS_subsample_ratio_list=[None]
        # beta_list = [1 / 2, 1, None]
        beta_list = [1]
        for subsample_ratio in ALS_subsample_ratio_list:
            print('!!! ALS subsample_ratio:', subsample_ratio)
            for beta in beta_list:
                print('!!! ALS initialized with beta:', beta)
                list_full_timed_errors = []
                iter1 = iter

                if subsample_ratio is not None:
                    iter1 = iter1

                for i in np.arange(num_repeat):
                    result_dict_ALS = ALS_run(X,
                                              n_components=n_components,
                                              iter=iter1,
                                              regularizer=0,
                                              # inverse regularizer on time mode (to promote long-lasting topics),
                                              # no regularizer on on words and tweets
                                              ini_loading=None,
                                              beta=beta,
                                              search_radius_const=search_radius_const,
                                              subsample_ratio=subsample_ratio,
                                              if_compute_recons_error=True,
                                              save_folder=save_folder,
                                              output_results=True)
                    time_error = result_dict_ALS.get('time_error')
                    list_full_timed_errors.append(time_error.copy())
                    # print('!!! list_full_timed_errors', len(list_full_timed_errors))

                    timed_errors_trials = np.asarray(
                        list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
                    result_dict_ALS.update({'timed_errors_trials': timed_errors_trials})

                    save_filename = "ALS_result_" + "beta_" + str(beta) + "_" + "subsample_" + str(subsample_ratio)
                    # np.save(save_folder + "/" + save_filename, result_dict_ALS)
                    ALS_result_list_dict.update({str(save_filename): result_dict_ALS})
                    np.save(save_folder + "/ALS_result_list_dict", ALS_result_list_dict)
                    print('ALS_result_list_dict.keys()', ALS_result_list_dict.keys())
                    result_dict_ALS = {}

    if run_MU:
        list_full_timed_errors = []
        print('!!! MU initialized')
        for i in np.arange(num_repeat):
            result_dict_MU = MU_run(X,
                                    n_components=n_components,
                                    iter=iter*1.5,
                                    regularizer=0,
                                    ini_loading=None,
                                    if_compute_recons_error=True,
                                    save_folder=save_folder,
                                    output_results=True)
            time_error = result_dict_MU.get('time_error')
            list_full_timed_errors.append(time_error.copy())
            # print('!!! list_full_timed_errors', len(list_full_timed_errors))

            timed_errors_trials = np.asarray(
                list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
            result_dict_MU.update({'timed_errors_trials': timed_errors_trials})

            np.save(save_folder + "/MU_result_" + str(file_name), result_dict_MU)
            print('result_dict_MU.keys()', result_dict_MU.keys())

    if run_OCPDL:
        print('!!! OCPDL initialized')
        list_full_timed_errors = []
        beta_list = [1]

        # SVD initialization using tensorly NCPD
        weights, factors = non_negative_parafac(X, rank=n_components, n_iter_max=1)
        ini_loading = {}
        ini_loading.update({"U0":factors[0]})
        ini_loading.update({"U1":factors[1]})

        for beta in beta_list:
            for i in np.arange(num_repeat):


                result_dict_OCPDL = OCPDL_run(X,
                                              n_components=n_components,
                                              iter=iter + 20,
                                              regularizer=1,
                                              batch_size=X.shape[-1]//10,
                                              beta=beta,
                                              ini_loading=ini_loading,
                                              mode_2be_subsampled=-1,
                                              if_compute_recons_error=False,
                                              save_folder=save_folder,
                                              output_results=True)

                time_error = result_dict_OCPDL.get('time_error')
                list_full_timed_errors.append(time_error.copy())
                print('!!! list_full_timed_errors', len(list_full_timed_errors))

                """
                timed_errors_trials = np.asarray(
                    list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
                result_dict_OCPDL.update({'timed_errors_trials': timed_errors_trials})

                np.save(save_folder + "/OCPDL_result_" + str(file_name), result_dict_OCPDL)
                """

                loading = result_dict_OCPDL.get("loading")
                factors = []
                for i in loading.keys():
                    factors.append(loading.get(str(i)))

                with open(os.path.join(save_folder,"ONCPD_factors_headlines.pickle"), "wb") as f:
                    pickle.dump(factors, f)

            print('result_dict_OCPDL.keys()', result_dict_OCPDL.keys())

    if plot_errors:
        save_filename = file_name + ".npy"
        #ALS_result_list_dict = np.load(save_folder + "/ALS_result_list_dict.npy", allow_pickle=True).item()
        OCPDL_result = np.load(save_folder + '/OCPDL_result_' + save_filename, allow_pickle=True).item()
        MU_result = np.load(save_folder + '/MU_result_' + save_filename, allow_pickle=True).item()
        # plot_benchmark_errors_list(ALS_result_list_dict=None, OCPDL_result, MU_result, name=str(save_filename), save_folder=save_folder)
        plot_benchmark_errors_list(OCPDL_result=OCPDL_result, MU_result=MU_result, name=str(save_filename), save_folder=save_folder)

    if plot_dictionary:
        save_filename = file_name + ".npy"
        #ALS_result_list_dict = np.load(save_folder + "/ALS_result_list_dict.npy", allow_pickle=True).item()
        OCPDL_result = np.load(save_folder + '/OCPDL_result_' + save_filename, allow_pickle=True).item()
        MU_result = np.load(save_folder + '/MU_result_' + save_filename, allow_pickle=True).item()




if __name__ == '__main__':
    main()
