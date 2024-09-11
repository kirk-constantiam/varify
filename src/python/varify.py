import pandas as pd
import torch
import pymc as pm
import numpy as np
import pytensor.tensor as pt
from sklearn.metrics import r2_score, roc_curve, auc, precision_recall_curve
import arviz as az
import matplotlib.pyplot as plt
import os
import seaborn as sns

RANDOM_SEED = 43
rng = np.random.default_rng(RANDOM_SEED)

def findlay_load_data(experiment = "X2"):
    df = pd.read_excel("data/brca1_findlay_2018.xlsx")
    experiment_df = df[df["experiment"] == experiment]
    df = experiment_df.reset_index()

    num_replicates = 2
    num_variants = len(df)

    library_counts = np.zeros([num_variants])
    endpoint_counts = np.zeros([num_replicates, num_variants])

    library_counts = np.array(df['library'])
    endpoint_counts[0,:] = np.array(df['d11.r1'])
    endpoint_counts[1,:] = np.array(df['d11.r2'])

    return library_counts, endpoint_counts, num_variants, df

def findlay_pymc_model(variants, endpoint_counts_shape, library_freq=None, library_amplitude=None, endpoint_amplitudes=None, library_counts=None, endpoint_counts=None, variant_effect=None):
    num_replicates, num_variants = endpoint_counts_shape

    coords = {
        'replicates': ['r1', 'r2'],
        'variants': variants,
    }

    with pm.Model(coords=coords) as pymc_model:
        if library_freq is None:
            library_freq = pm.Dirichlet('library_freq', a=np.ones((num_variants,)), observed=library_freq)

        if library_amplitude is None:
            if library_counts is None:
                # library_amplitude = pm.Uniform('library_amplitude', lower=0, upper=400_000)
                library_amplitude = pm.Normal('library_amplitude', mu=350_000, sigma=50_000)
            else:
                library_amplitude = library_counts.sum()

        if endpoint_amplitudes is None:
            if endpoint_counts is None:
                # endpoint_amplitudes = pm.Uniform('endpoint_amplitudes', lower=0, upper=400_000, shape=(num_replicates,))
                # endpoint_amplitudes = pm.Normal('endpoint_amplitudes', mu=350_000, sigma=50_000, shape=(num_replicates,))
                endpoint_amplitudes = pm.Normal('endpoint_amplitudes', mu=350_000, sigma=50_000, dims='replicates')
            else:
                endpoint_amplitudes = endpoint_counts.sum(axis=1)

        library_counts = pm.Poisson('library_counts', mu=library_amplitude*library_freq, shape=(num_variants,), observed=library_counts)

        if variant_effect is None:
            # variant_effect = pm.Normal('variant_effect', mu=0, sigma=10, shape=(num_variants,))
            variant_effect = pm.Normal('variant_effect', mu=0, sigma=10, dims='variants')

        # Calculate the expected value of endpoint_counts
        # mu_endpoint_counts = (endpoint_amplitudes[:, None] * (library_freq * 2 ** variant_effect)[None, :]).T
        mu_endpoint_counts = endpoint_amplitudes[:, None] * (library_freq * pt.pow(2.0, variant_effect))[None, :]     

        # endpoint_counts = pm.Poisson('endpoint_counts', mu=mu_endpoint_counts, shape=(num_replicates, num_variants), observed=endpoint_counts)
        endpoint_counts = pm.Poisson('endpoint_counts', mu=mu_endpoint_counts, dims=('replicates', 'variants'), observed=endpoint_counts)

    return pymc_model

def findlay_simulate_variant_effects(num_variants):
    # Step 1: Specify a set of known variant effects
    known_variant_effects = np.random.normal(0, 1, size=num_variants)
    return known_variant_effects

def visualize_pymc_model(pymc_model):
    pm.model_to_graphviz(pymc_model)

def findlay_simulate_quantities(variants, library_counts, endpoint_counts, num_variants):
    known_variant_effects = findlay_simulate_variant_effects(num_variants)

    pymc_sge_model = findlay_pymc_model(
        variants,
        endpoint_counts.shape, 
        library_freq=None, 
        library_amplitude=None, 
        endpoint_amplitudes=None, 
        library_counts=None, 
        endpoint_counts=None, 
        variant_effect=known_variant_effects
    )

    # Step 2: Simulate count data from the model using these known effects
    # library_freq=None, library_amplitude=None, endpoint_amplitudes=None, library_counts=None, endpoint_counts=None
    with pymc_sge_model as prior_model:    
        # Simulate the data
        simulated_data = pm.sample_prior_predictive(samples=100, var_names=[
            'library_freq', 
            'library_amplitude',
            'endpoint_amplitudes',
            'library_counts', 
            'endpoint_counts',
        ])

    return known_variant_effects, simulated_data

def findlay_infer_variant_effects_from_simulated_data(variants, simulated_data, endpoint_counts):
    pymc_sge_model = findlay_pymc_model(
        variants,
        endpoint_counts.shape, 
        library_freq=simulated_data.prior.get('library_freq').to_numpy().mean(axis=(0,1)), 
        library_amplitude=simulated_data.prior.get('library_amplitude').to_numpy().mean(axis=(0,1)), 
        endpoint_amplitudes=simulated_data.prior.get('endpoint_amplitudes').to_numpy().mean(axis=(0,1)), 
        library_counts=simulated_data.prior.get('library_counts').to_numpy().mean(axis=(0,1)), 
        endpoint_counts=simulated_data.prior.get('endpoint_counts').to_numpy().mean(axis=(0,1)), 
        variant_effect=None
    )

    # At this point, `simulated_data` is a dictionary with keys 'library_counts' and 'endpoint_counts'
    # Each key corresponds to a 100 x num_variants or 100 x num_replicates x num_variants array of simulated count data

    # Step 3: Fit the model to the simulated data
    with pymc_sge_model as posterior_model:
        # Perform the inference
        trace = pm.sample(2000, tune=1000, random_seed=rng)

    # Step 4: Compare the inferred effects to the known effects
    # Extract the posterior samples for 'variant_effect'
    inferred_variant_effects = trace.posterior.get('variant_effect')

    return inferred_variant_effects.mean(axis=(0,1))

def findlay_compare_known_and_inferred_variant_effects(known_variant_effects, inferred_variant_effects):
    import matplotlib.pyplot as plt
    # Make a comparison plot
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(known_variant_effects, inferred_variant_effects, label='Inferred vs True')
    plt.plot(known_variant_effects, known_variant_effects, color='red', label='Ideal')
    plt.xlabel('True variant effect')
    plt.ylabel('Inferred variant effect')
    plt.legend()
    
    return fig

def compute_r2(y_true, y_pred):
    # Calculate R2 score
    r2 = r2_score(y_true, y_pred)

    print(f'R2 score: {r2}')

    return r2

def findlay_infer_variant_effects_from_real_data(variants, library_counts, endpoint_counts):
    pymc_sge_model = findlay_pymc_model(
        variants,
        endpoint_counts.shape, 
        library_freq=None, 
        library_amplitude=None, 
        endpoint_amplitudes=None, 
        library_counts=library_counts, 
        endpoint_counts=endpoint_counts, 
        variant_effect=None
    )

    with pymc_sge_model:
        idata = pm.sample(1000, tune=2000, random_seed=rng)

    return idata, pymc_sge_model  

def perform_posterior_predictive_checks(idata, pymc_model):
    with pymc_model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

    return idata
    
def findlay_compare_original_and_inferred_variant_effects(df, idata):
    import matplotlib.pyplot as plt
    variant_effects_fig = plt.figure()
    plt.scatter((df['d11/lib.raw.r1']+df['d11/lib.raw.r2'])/2., idata.posterior.get('variant_effect').mean(axis=(0,1)), alpha=0.4)
    library_freqs_fig = plt.figure()
    plt = plt.scatter(df['library.freq'], idata.posterior.get('library_freq').mean(axis=(0,1)), alpha=0.4)
    return variant_effects_fig, library_freqs_fig

def findlay_compare_metrics(df, idata, prior=0.1):
    assay_df = df
    pathogenic_df = assay_df[assay_df['clinvar_simple'].isin(['Pathogenic', 'Likely pathogenic'])]
    pathogenic_scores = np.array(pathogenic_df.apply(lambda x: (x['d11/lib.raw.r1']+x['d11/lib.raw.r2'])/2., axis=1))
    num_pathogenic_total = len(pathogenic_scores)
    benign_df = assay_df[assay_df['clinvar_simple'].isin(['Benign', 'Likely benign'])]
    benign_scores = np.array(benign_df.apply(lambda x: (x['d11/lib.raw.r1']+x['d11/lib.raw.r2'])/2., axis=1))
    num_benign_total = len(benign_scores)
    num_reference_total = num_pathogenic_total + num_benign_total
    all_reference_scores = np.concatenate((pathogenic_scores, benign_scores))

    pathogenic_binaries = [1]*len(pathogenic_scores)
    benign_binaries = [0]*len(benign_scores)
    binaries_list = pathogenic_binaries+benign_binaries

    fpr, tpr, thresholds = roc_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    original_auroc = auc(fpr,tpr)
    print(f'Original AUROC: {original_auroc}')

    precision, recall, _ = precision_recall_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    # Compute Area Under the Precision-Recall Curve
    original_auprc = auc(recall, precision)
    print(f'Original AUPRC: {original_auprc}')

    # Compute AUBPRC
    prior = 0.1
    original_aubprc = (original_auprc*(1-prior))/(original_auprc*(1-prior)+(1-original_auprc)*prior)
    print(f'Original AUBPRC: {original_aubprc}')

    df.loc[:,'varify_inferred_score'] = idata.posterior.get('variant_effect').mean(axis=(0,1))

    assay_df = df
    pathogenic_df = assay_df[assay_df['clinvar_simple'].isin(['Pathogenic', 'Likely pathogenic'])]
    pathogenic_scores = np.array(pathogenic_df['varify_inferred_score'])
    num_pathogenic_total = len(pathogenic_scores)
    benign_df = assay_df[assay_df['clinvar_simple'].isin(['Benign', 'Likely benign'])]
    benign_scores = np.array(benign_df['varify_inferred_score'])
    num_benign_total = len(benign_scores)
    num_reference_total = num_pathogenic_total + num_benign_total
    all_reference_scores = np.concatenate((pathogenic_scores, benign_scores))

    pathogenic_binaries = [1]*len(pathogenic_scores)
    benign_binaries = [0]*len(benign_scores)
    binaries_list = pathogenic_binaries+benign_binaries

    fpr, tpr, thresholds = roc_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    varify_auroc = auc(fpr,tpr)
    print(f'Varify AUROC: {varify_auroc}')

    precision, recall, _ = precision_recall_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    # Compute Area Under the Precision-Recall Curve
    varify_auprc = auc(recall, precision)
    print(f'Varify AUPRC: {varify_auprc}')

    # Compute AUBPRC
    varify_aubprc = (varify_auprc*(1-prior))/(varify_auprc*(1-prior)+(1-varify_auprc)*prior)
    print(f'Varify AUBPRC: {varify_aubprc}')

    return original_auroc, original_auprc, original_aubprc, varify_auroc, varify_auprc, varify_aubprc

def load_clinvar_df(path="data/clinvar_annotations.csv"):
    clinvar_df = pd.read_csv(path)
    return clinvar_df

def load_matreyek_df():
    replicates = ['e1s1', 'e1s2', 'e1s3', 'e1s4'] #, 'e3s1', 'e3s2', 'e3s3']

    dfs = []
    total_df = pd.DataFrame()
    for replicate in replicates:

        # Path to the CSV file
        filepath = f"./data/pten_matreyek_2021/{replicate}_codon_weighted_ave.csv"

        data = []
        with open(filepath, 'r') as f:
            header = next(f).strip().split(',')
            num_columns = len(header)

            for line in f:
                row = line.strip().split(',')
                first_column = ','.join(row[:-num_columns+1])
                rest_of_row = row[-num_columns+1:]
                row = [first_column] + rest_of_row
                data.append(row)

        df = pd.DataFrame(data, columns=header)
        df['replicate'] = replicate

        if replicate in ['e1s1', 'e1s2', 'e1s3', 'e1s4']:
            for column in ['b1a', 'b1b', 'b2a', 'b2b', 'b3a', 'b3b', 'b4a', 'b4b']:
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
            df['b1'] = df['b1a'] + df['b1b']
            df['b2'] = df['b2a'] + df['b2b']
            df['b3'] = df['b3a'] + df['b3b']
            df['b4'] = df['b4a'] + df['b4b']
        for column in ['b1', 'b2', 'b3', 'b4', 'mean_freq', 'weighted_ave']:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(float)

        # Now, df is a DataFrame containing the data from the CSV file
        # print(len(df))
        dfs.append(df)
        # print(replicate, len(df))

        
    total_df = pd.concat(dfs)


    
    # summed_df = total_df[['replicate', 'variant', 'b1a', 'b1b', 'b2a', 'b2b', 'b3a', 'b3b', 'b4a', 'b4b']].groupby(['replicate', 'variant']).sum()
    summed_df = total_df[['replicate', 'variant', 'b1', 'b2', 'b3', 'b4']].groupby(['replicate', 'variant']).sum()
    # summed_df['variant'] = summed_df.index
    # summed_df.reset_index(inplace=True)


    # summed_df = summed_df[['b1', 'b2', 'b3', 'b4']]

    # Calculate the fraction of each variant observed in each bin
    summed_df.loc[:, 'f_b1'] = summed_df['b1']/summed_df['b1'].sum()
    summed_df.loc[:, 'f_b2'] = summed_df['b2']/summed_df['b2'].sum()
    summed_df.loc[:, 'f_b3'] = summed_df['b3']/summed_df['b3'].sum()
    summed_df.loc[:, 'f_b4'] = summed_df['b4']/summed_df['b4'].sum()

    # Calculate the fraction of each variant observed in total
    summed_df.loc[:, 'f_total'] = summed_df[['b1', 'b2', 'b3', 'b4']].sum(axis=1)/summed_df[['b1', 'b2', 'b3', 'b4']].sum().sum()

    summed_df = summed_df[summed_df.apply(lambda v: v['f_b1'] + v['f_b2'] + v['f_b3'] + v['f_b4'] != 0, axis=1)]

    # Calculate weighted average fraction for each variant
    summed_df.loc[:, 'w_v'] = summed_df.apply(lambda v: (v['f_b1']*0.25 + v['f_b2']*0.50 + v['f_b3']*0.75 + v['f_b4']*1.00)/(v['f_b1'] + v['f_b2'] + v['f_b3'] + v['f_b4']), axis=1)

    # summed_df = summed_df[['variant', 'b1', 'b2', 'b3', 'b4']].groupby(['variant', 'replicate']).sum().astype(int)

    return summed_df

def matreyek_pymc_model(
    variants,
    fluorescence_ratio_distribution_mu_variant=None, 
    fluorescence_ratio_distribution_sigma_variant=None,
    read_depths=None,
    bin_counts=None,
):
    coords = {
        "variant": variants,
        "bin": [1, 2, 3, 4],
    }

    if type(bin_counts) == pd.DataFrame:
        bin_counts = bin_counts[['b1', 'b2', 'b3', 'b4']].values
    
    with pm.Model(coords=coords) as model:
        num_variants = len(coords['variant'])

        if fluorescence_ratio_distribution_mu_variant is None:        
            normal_dist = pm.Normal.dist(mu=0.5, sigma=0.25)    
            fluorescence_ratio_distribution_mu_variant = pm.Truncated(
                "fluorescence_ratio_distribution_mu_variant", 
                normal_dist,
                lower=0.0, 
                upper=2.0, 
                dims=('variant')
            )

        if fluorescence_ratio_distribution_sigma_variant is None:
            fluorescence_ratio_distribution_sigma_variant = pm.HalfNormal(
                "fluorescence_ratio_distribution_sigma_variant", 
                sigma=0.25, 
                dims=('variant')
            )

        # bin_cutpoints = pm.Normal(
        #     "bin_cutpoints", 
        #     mu=[0.25, 0.5, 0.75], 
        #     sigma=0.25, 
        #     size=3, 
        #     transform=pm.distributions.transforms.univariate_ordered
        # )
        bin_cutpoints = np.array([0.25, 0.5, 0.75])
        bin_cutpoints_reshaped = bin_cutpoints[:, np.newaxis]
        
        cdf_values = 0.5 * (1 + pt.erf((bin_cutpoints_reshaped - fluorescence_ratio_distribution_mu_variant) / (fluorescence_ratio_distribution_sigma_variant * pt.sqrt(2))))
        cdf_values = pt.concatenate([cdf_values, np.expand_dims(np.ones(num_variants), axis=0)])
        cdf_values = pt.concatenate([np.expand_dims(np.zeros(num_variants), axis=0), cdf_values])
        bin_fractions = pt.extra_ops.diff(cdf_values, axis=0)

        if bin_counts is None:
            read_depth = read_depths
        else:
            read_depth = bin_counts.sum(axis=1)

        if bin_counts is None:
            bin_counts = pm.Poisson(
                "bin_counts",
                mu=bin_fractions*read_depth,
                dims=('bin', 'variant'),
            )
        else:
            bin_counts = pm.Poisson(
                "bin_counts",
                mu=bin_fractions*read_depth,
                dims=('bin', 'variant'),
                observed=bin_counts.T,
            )
        
        # idata = pm.sample_prior_predictive()
        # idata = pm.sample(target_accept=0.9)
    return model

def matreyek_simulate_variant_effects(num_variants):
    from scipy.stats import truncnorm, halfnorm
    # num_variants = len(summed_replicate_df)
    loc = 0.5
    scale = 0.25
    a, b = (0 - loc) / scale, (2 - loc) / scale
    # Step 1: Specify a set of known variant effects
    fluorescence_ratio_distribution_mu_variant = truncnorm.rvs(a, b, loc=loc, scale=scale, size=num_variants)

    # fluorescence_ratio_distribution_sigma_variant = halfnorm.rvs(loc=0, scale=0.25, size=num_variants)
    loc = 0.2
    scale = 0.25
    a, b = (0.1 - loc) / scale, (1 - loc) / scale
    # Step 1: Specify a set of known variant effects
    fluorescence_ratio_distribution_sigma_variant = truncnorm.rvs(a, b, loc=loc, scale=scale, size=num_variants)
    
    return fluorescence_ratio_distribution_mu_variant, fluorescence_ratio_distribution_sigma_variant

def matreyek_simulate_bin_counts(variants, read_depths):
    num_variants = len(variants)
    fluorescence_ratio_distribution_mu_variant, fluorescence_ratio_distribution_sigma_variant = matreyek_simulate_variant_effects(num_variants)

    pymc_vampseq_model = matreyek_pymc_model(
        variants,
        fluorescence_ratio_distribution_mu_variant=fluorescence_ratio_distribution_mu_variant, 
        fluorescence_ratio_distribution_sigma_variant=fluorescence_ratio_distribution_sigma_variant,
        read_depths=read_depths,
        bin_counts=None,
    )

    # Step 2: Simulate count data from the model using these known effects
    # library_freq=None, library_amplitude=None, endpoint_amplitudes=None, library_counts=None, endpoint_counts=None
    with pymc_vampseq_model as prior_model:    
        # Simulate the data
        simulated_data = pm.sample_prior_predictive(samples=100, var_names=[
            # 'fluorescence_ratio_distribution_mu_variant', 
            # 'fluorescence_ratio_distribution_sigma_variant',
            'bin_counts',
        ])

    return fluorescence_ratio_distribution_mu_variant, fluorescence_ratio_distribution_sigma_variant, simulated_data

def matreyek_infer_variant_effects_from_simulated_data(variants, simulated_data, nuts_sampler='pymc'):
    pymc_vampseq_model = matreyek_pymc_model(
        variants,
        fluorescence_ratio_distribution_mu_variant=None, 
        fluorescence_ratio_distribution_sigma_variant=None,
        read_depths=None,
        bin_counts=simulated_data['prior']['bin_counts'].to_numpy().mean(axis=(0,1)).T,
    )

    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

    # Step 3: Fit the model to the simulated data
    with pymc_vampseq_model as posterior_model:
        # Perform the inference
        # if nuts_sampler == 'blackjax':
        #     chains = 1
        # else:
        #     chains = 4
        chains = 4
        trace = pm.sample(
            2000, 
            tune=1000, 
            random_seed=rng, 
            nuts_sampler=nuts_sampler, 
            chains=chains, 
            # nuts_sampler_kwargs={
            #     #'chain_method': 'vectorized',
            #     'postprocessing_backend': 'cpu',
            # }
        )

    # Step 4: Compare the inferred effects to the known effects
    # Extract the posterior samples for 'variant_effect'
    # inferred_fluorescence_ratio_distribution_mu_variant = trace.posterior.get('fluorescence_ratio_distribution_mu_variant')
    # inferred_fluorescence_ratio_distribution_sigma_variant = trace.posterior.get('fluorescence_ratio_distribution_sigma_variant')

    return trace #, inferred_fluorescence_ratio_distribution_mu_variant.mean(axis=(0,1)), inferred_fluorescence_ratio_distribution_sigma_variant.mean(axis=(0,1))

# def matreyek_compare_known_and_inferred_variant_effects(
#     fluorescence_ratio_distribution_mu_variant, 
#     inferred_fluorescence_ratio_distribution_mu_variant,
#     fluorescence_ratio_distribution_sigma_variant,
#     inferred_fluorescence_ratio_distribution_sigma_variant,
# ):
#     import matplotlib.pyplot as plt
#     # Make a comparison plot
#     mu_fig = plt.figure(figsize=(8, 8))
#     plt.scatter(fluorescence_ratio_distribution_mu_variant, inferred_fluorescence_ratio_distribution_mu_variant, label='Inferred vs True', alpha=0.2)
#     plt.xlabel('True mean variant effect')
#     plt.ylabel('Inferred mean variant effect')
#     plt.legend()

#     sigma_fig = plt.figure(figsize=(8, 8))
#     plt.scatter(fluorescence_ratio_distribution_sigma_variant, inferred_fluorescence_ratio_distribution_sigma_variant, label='Inferred vs True', alpha=0.2)
#     plt.xlabel('True std variant effect')
#     plt.ylabel('Inferred std variant effect')
#     plt.legend()
    
#     return mu_fig, sigma_fig

# def matreyek_compare_known_and_inferred_variant_effects(
#     trace,
#     fluorescence_ratio_distribution_mu_variant, 
#     fluorescence_ratio_distribution_sigma_variant,
# ):

#     inferred_fluorescence_ratio_distribution_mu_variant = trace.posterior.get('fluorescence_ratio_distribution_mu_variant').mean(axis=(0,1))
#     inferred_fluorescence_ratio_distribution_sigma_variant = trace.posterior.get('fluorescence_ratio_distribution_sigma_variant').mean(axis=(0,1))

#     # Define a general style
#     sns.set_style('whitegrid')
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['axes.labelsize'] = 16
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14
#     plt.rcParams['legend.fontsize'] = 14

#     # Make a comparison plot for means
#     mu_fig, mu_ax = plt.subplots(figsize=(8, 8))
#     mu_ax.scatter(fluorescence_ratio_distribution_mu_variant, inferred_fluorescence_ratio_distribution_mu_variant, label='Inferred vs True', alpha=0.5, s=20)
#     mu_ax.plot([min(fluorescence_ratio_distribution_mu_variant), max(fluorescence_ratio_distribution_mu_variant)], 
#                [min(fluorescence_ratio_distribution_mu_variant), max(fluorescence_ratio_distribution_mu_variant)], 
#                color='red')
#     mu_ax.set_xlabel('True mean variant effect')
#     mu_ax.set_ylabel('Inferred mean variant effect')
#     mu_ax.set_title('Comparison of True and Inferred Mean Variant Effects')
#     mu_ax.set_aspect('equal', 'box')
#     mu_ax.legend()

#     # # Make a comparison plot for standard deviations
#     # sigma_fig, sigma_ax = plt.subplots(figsize=(8, 8))
#     # sigma_ax.scatter(fluorescence_ratio_distribution_sigma_variant, inferred_fluorescence_ratio_distribution_sigma_variant, label='Inferred vs True', alpha=0.5, s=20)
#     # sigma_ax.plot([min(fluorescence_ratio_distribution_sigma_variant), max(fluorescence_ratio_distribution_sigma_variant)], 
#     #               [min(fluorescence_ratio_distribution_sigma_variant), max(fluorescence_ratio_distribution_sigma_variant)], 
#     #               color='red')
#     # sigma_ax.set_xlabel('True std variant effect')
#     # sigma_ax.set_ylabel('Inferred std variant effect')
#     # sigma_ax.set_title('Comparison of True and Inferred Std Variant Effects')
#     # sigma_ax.set_aspect('equal', 'box')
#     # sigma_ax.legend()

#     return mu_fig #, sigma_fig

def matreyek_compare_known_and_inferred_variant_effects(
    trace,
    fluorescence_ratio_distribution_mu_variant, 
    fluorescence_ratio_distribution_sigma_variant,
    hdi_size=50,
):
    inferred_fluorescence_ratio_distribution_mu_variant = trace.posterior.get('fluorescence_ratio_distribution_mu_variant').mean(axis=(0,1))

    # Compute the HDI of the inferred mu variant
    lower_bound = (100 - hdi_size) / 2
    upper_bound = (100 + hdi_size) / 2
    hdi_mu_variant = np.percentile(trace.posterior.get('fluorescence_ratio_distribution_mu_variant'), [lower_bound, upper_bound], axis=(0,1))
    hdi_mu_variant_lower = np.abs(inferred_fluorescence_ratio_distribution_mu_variant - hdi_mu_variant[0])
    hdi_mu_variant_upper = np.abs(hdi_mu_variant[1] - inferred_fluorescence_ratio_distribution_mu_variant)

    # Define a general style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # Make a comparison plot for means
    mu_fig, mu_ax = plt.subplots(figsize=(3, 3))
    mu_ax.errorbar(fluorescence_ratio_distribution_mu_variant, inferred_fluorescence_ratio_distribution_mu_variant, 
                yerr=[hdi_mu_variant_lower, hdi_mu_variant_upper], 
                fmt='o', color='gray', alpha=0.2, label=None, zorder=1)  # Plot error bars with lower alpha
    mu_ax.scatter(fluorescence_ratio_distribution_mu_variant, inferred_fluorescence_ratio_distribution_mu_variant,
                color='blue', alpha=0.2, label='Inferred vs True', zorder=2)  # Plot data points with higher alpha
    mu_ax.plot([min(fluorescence_ratio_distribution_mu_variant), max(fluorescence_ratio_distribution_mu_variant)], 
               [min(fluorescence_ratio_distribution_mu_variant), max(fluorescence_ratio_distribution_mu_variant)], 
               color='red', zorder=3)
    mu_ax.set_xlabel('True variant effect')
    mu_ax.set_ylabel('Inferred variant effect')
    mu_ax.set_title('True vs. Inferred Variant Effects', fontsize=12)
    mu_ax.set_aspect('equal', 'box')
    # mu_ax.legend()

    return mu_fig

def matreyek_infer_variant_effects_from_real_data(variants, bin_counts):
    pymc_vampseq_model = matreyek_pymc_model(
        variants,
        fluorescence_ratio_distribution_mu_variant=None, 
        fluorescence_ratio_distribution_sigma_variant=None,
        read_depths=None,
        bin_counts=bin_counts,
    )

    with pymc_vampseq_model:
        idata = pm.sample(1000, tune=2000, random_seed=rng)

    return idata, pymc_vampseq_model

def matreyek_compare_scores_using_clinvar_annotations(idata, summed_replicate_df):
    clinvar_df = load_clinvar_df()
    pten_clinvar_df = clinvar_df[clinvar_df['gene'] == 'PTEN']
    summed_replicate_df['aa_variant'] = 'p.' + summed_replicate_df.index.astype(str)
    summed_replicate_df = summed_replicate_df.merge(pten_clinvar_df, on='aa_variant', how='left')
    summed_replicate_df['variant_effect'] = idata['posterior']['fluorescence_ratio_distribution_mu_variant'].mean(axis=(0,1))

    assay_df = summed_replicate_df
    pathogenic_df = assay_df[assay_df['clinical_annotation'].isin(['Pathogenic'])]
    pathogenic_scores = np.array(pathogenic_df['w_v'])
    num_pathogenic_total = len(pathogenic_scores)
    benign_df = assay_df[assay_df['clinical_annotation'].isin(['Benign'])]
    benign_scores = np.array(benign_df['w_v'])
    num_benign_total = len(benign_scores)
    num_reference_total = num_pathogenic_total + num_benign_total
    all_reference_scores = np.concatenate((pathogenic_scores, benign_scores))

    pathogenic_binaries = [1]*len(pathogenic_scores)
    benign_binaries = [0]*len(benign_scores)
    binaries_list = pathogenic_binaries+benign_binaries

    fpr, tpr, thresholds = roc_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    original_auroc = auc(fpr,tpr)
    print(f'Original AUROC: {original_auroc}')

    precision, recall, _ = precision_recall_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    # Compute Area Under the Precision-Recall Curve
    original_auprc = auc(recall, precision)
    print(f'Original AUPRC: {original_auprc}')

    # Compute AUBPRC
    prior = 0.1
    original_aubprc = (original_auprc*(1-prior))/(original_auprc*(1-prior)+(1-original_auprc)*prior)
    print(f'Original AUBPRC: {original_aubprc}')

    assay_df = summed_replicate_df
    pathogenic_df = assay_df[assay_df['clinical_annotation'].isin(['Pathogenic'])]
    pathogenic_scores = np.array(pathogenic_df['variant_effect'])
    num_pathogenic_total = len(pathogenic_scores)
    benign_df = assay_df[assay_df['clinical_annotation'].isin(['Benign'])]
    benign_scores = np.array(benign_df['variant_effect'])
    num_benign_total = len(benign_scores)
    num_reference_total = num_pathogenic_total + num_benign_total
    all_reference_scores = np.concatenate((pathogenic_scores, benign_scores))

    pathogenic_binaries = [1]*len(pathogenic_scores)
    benign_binaries = [0]*len(benign_scores)
    binaries_list = pathogenic_binaries+benign_binaries

    fpr, tpr, thresholds = roc_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    varify_auroc = auc(fpr,tpr)
    print(f'Varify AUROC: {varify_auroc}')

    precision, recall, _ = precision_recall_curve(binaries_list, -1*np.array(all_reference_scores), pos_label=1)
    # Compute Area Under the Precision-Recall Curve
    varify_auprc = auc(recall, precision)
    print(f'Varify AUPRC: {varify_auprc}')

    # Compute AUBPRC
    varify_aubprc = (varify_auprc*(1-prior))/(varify_auprc*(1-prior)+(1-varify_auprc)*prior)
    print(f'Varify AUBPRC: {varify_aubprc}')

    return original_auroc, original_auprc, original_aubprc, varify_auroc, varify_auprc, varify_aubprc