# %%
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
pd.set_option("display.max_columns", 50)

# Update the function to clean the data
def clean_data(csv_file):
    # Load the csv file
    df = pd.read_csv(csv_file)
    
    # Rename columns to more manageable names
    df = df.rename(columns={'Please enter your full name (First & Last)': 'doctor', 'Please enter the case number': 'pt'})
    
    # Standardize doctor names: make them lowercase
    df['doctor'] = df['doctor'].str.lower()
    
    # Fix misspellings of doctor names
    df['doctor'] = df['doctor'].replace({
        'aaron pula': 'aaron paul',
        'vanes carlota andreu arasa': 'vanesa carlota andreu arasa',
        'vanesa carlota andreu': 'vanesa carlota andreu arasa',
        'vanesa carlotaandreu arasa': 'vanesa carlota andreu arasa',
        # Add more replacements here if necessary
    })
    #cleaned_data['doctor'] = cleaned_data['doctor'].replace('65', 'asim mian')

    # Standardize case numbers: remove 'case' (with and without a trailing space) and convert to integers where possible
    df['pt'] = df['pt'].str.replace('case ', '', case=False).str.replace('case', '', case=False).str.strip()  # Remove 'case' prefix (with and without space) and trailing spaces
    df['pt'] = pd.to_numeric(df['pt'], errors='coerce')  # Convert to numeric, set non-numeric values to NaN
    
    # Drop rows where 'pt' is NaN
    df = df.dropna(subset=['pt'])
    
    # Convert 'pt' to int (now safe to do so since we removed non-numeric values)
    df['pt'] = df['pt'].astype(int)
    
    return df

def missing_cases_for_doctor(dataframe, doctor_name):
    # Get the case numbers for the given doctor
    doctor_cases = set(dataframe[dataframe['doctor'] == doctor_name]['pt'])

    # Define the full set of case numbers
    full_case_set = set(range(1, 71))  # Assuming case numbers are from 1 to 70

    # Find the missing cases for the given doctor
    missing_cases = full_case_set - doctor_cases

    return missing_cases


# Clean the new csv data
cleaned_data = clean_data('~/RadiologistRatings/70_test_cases/ADRDRadiologistTask_DATA_2023-08-12_1555.csv')

# print(cleaned_data)

radiologist_data_dict_df = pd.read_csv("~/RadiologistRatings/70_test_cases/ADRDRadiologistTask_DataDictionary_2023-09-14.csv")

# Function to convert a string to a dictionary
def str_to_dict(entry):
    if not isinstance(entry, str):
        return entry
    result = {}
    segments = entry.split("|")
    for segment in segments:
        segment = segment.strip()
        if "," in segment:
            key, value = map(str.strip, segment.split(","))
            result[key] = value
        else:
            return "slider"
    return result

# print(radiologist_data_dict_df)
region_columns = list(cleaned_data.iloc[:, 7:-34].columns)
print(region_columns)
region_names = []
for i in range(len(region_columns)):
    name = radiologist_data_dict_df[radiologist_data_dict_df['Variable / Field Name'] == region_columns[i]]['Field Label'].iloc[0]
    region_names.append(name)
# print(region_names)

columns_to_rename = cleaned_data.columns[7:-34]

# Make sure the lengths of columns to rename and new names are the same
if len(columns_to_rename) != len(region_names):
    raise ValueError("Mismatch between the number of columns to rename and the new column names provided.")

# Update the DataFrame with the new column names
cleaned_data.columns = cleaned_data.columns[:7].tolist() + region_names + cleaned_data.columns[-34:].tolist()

cleaned_data.head()  # To verify the changes

binary_region_questions = {
    "Atrophy/volume loss in the frontal lobe": ['Frontal lobe (general)', 'Left anterior insula',
       'Right anterior insula', 'Left anterior cingulate gyrus',
       'Right Anterior cingulate gyrus', 'Left precentral gyrus', 'Right precentral gyrus',
       'Left caudate nucleus', 'Right caudate nucleus'],
    "Is there mesial temporal lobe atrophy?": ['Mesial Temporal lobe atrophy (rating) ','Left mesial temporal lobe',
       'Right mesial temporal lobe', 'Left hippocampus', 'Right hippocampus',
       'Left amygdala', 'Right amygdala', 'Atrophy of left parahippocampus?',
       'Atrophy of right parahippocampus?'],
    "Is there non-mesial temporal lobe atrophy?": ['Non-Mesial Temporal Lobe Atrophy (rating)', 'Left temporal lobe',
       'Right temporal lobe', 'Left lateral temporal lobe',
       'Right lateral temporal lobe', 'Anterior temporal lobe',
       'Posterior temporal lobe', 'Left fusiform gyrus',
       'Right fusiform gyrus', 'Left middle and inferior temporal gyrus',
       'Right middle and  inferior temporal gyrus'],
    "Is there parietal lobe atrophy?": ['left parietal lobe atrophy (general)',
       'right parietal lobe atrophy (general)',],
    "Is there occipital lobe atrophy?": ['Atrophy in the occipital lobe'],
}

for binary_col, regions in binary_region_questions.items():
    if binary_col in cleaned_data.columns:
        cleaned_data.loc[cleaned_data[binary_col] == 0, regions] = cleaned_data[regions].fillna(1)

print(cleaned_data.columns)
# save to csv
cleaned_data.to_csv('~/RadiologistRatings/70_test_cases/cleaned_radiologist_data.csv', index=False)
# create one dataframe that contains the aggregated data for each case across the 7 doctors using the mode

# Group by case number and aggregate the data

def get_mode(series):
    return series.mode()[0] if not series.mode().empty else None

consensus_rating_df = cleaned_data.groupby('pt').agg(get_mode)
# print(consensus_rating_df)

diagnosis_cases = pd.read_csv("~/RadiologistRatings/70_test_cases/clinician_review_cases_converted_radio.csv")
# print(diagnosis_cases.columns)
diagnosis_cases['pt'] = diagnosis_cases['case_number'].str.replace('CASE_', '').str.strip().astype(int)
confidence_columns = ['alz_ds', 'lbd_lb_pdd', 'vasdem', 'prion_d', 'ftld_v', 'nph_2']
etiology_columns = ['AD_lb', 'LBD_lb', 'VD_lb', 'PRD_lb', 'FTD_lb', 'NPH_lb']
# %%
model_probabilities = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_test_data_AD_nAD(FTD,VD,PRD)_pred.csv')
etiologies = ['AD', 'nAD']

# model_probabilities = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_test_data_AD_VD_FTD_PRD_pred.csv')
# etiologies = ['AD', 'VD', 'FTD', 'PRD']
# the NACCID must be looked up in the diagnosis_cases dataframe to get the pt
model_probabilities = model_probabilities.merge(diagnosis_cases[['ID', 'pt']], left_on='NACCID', right_on='ID', how='left')
print(model_probabilities)
# set index to pt
model_probabilities.set_index('pt', inplace=True)
model_probabilities.rename(columns={
    'AD': 'AD_model',
    'nAD': 'nAD_model',
    'VD': 'VD_model',
    'FTD': 'FTD_model',
    'PRD': 'PRD_model',
    'LBD': 'LBD_model',
}, inplace=True)
model_probabilities.head()

# %%
# method_of_nAD = 'Prod_nAD'
method_of_nAD = 'Max_nAD'


def inclusion_exclusion_combination(df, component_columns, combined_column):
    def inclusion_exclusion(confidences):
        n = len(confidences)
        combined_conf = 0
        for k in range(1, n+1):
            sign = (-1) ** (k + 1)
            for i in range(1 << n):
                if bin(i).count('1') == k:
                    product = 1
                    for j in range(n):
                        if i & (1 << j):
                            product *= confidences[j]
                    combined_conf += sign * product
        return combined_conf
    
    df[combined_column] = df.apply(lambda row: inclusion_exclusion([row[col] for col in component_columns]), axis=1)
    return df

def product_based_combination(df, component_columns, combined_column):
    def product_based(confidences):
        product = 1
        for c in confidences:
            product *= (1 - c)
        return 1 - product
    
    df[combined_column] = df.apply(lambda row: product_based([row[col] for col in component_columns]), axis=1)
    return df

def max_confidence(df, component_columns, combined_column):
    df[combined_column] = df[component_columns].max(axis=1)
    return df

cols_to_combine = ['vasdem', 'prion_d', 'ftld_v']
cleaned_data = inclusion_exclusion_combination(cleaned_data, cols_to_combine, 'IncExc_nAD')
cleaned_data = product_based_combination(cleaned_data, cols_to_combine, 'Prod_nAD')
cleaned_data = max_confidence(cleaned_data, cols_to_combine, 'Max_nAD')
cleaned_data.rename(columns={
    'alz_ds': 'AD_rad',
    'lbd_lb_pdd': 'LBD_rad',
    'vasdem': 'VD_rad',
    'prion_d': 'PRD_rad',
    'ftld_v': 'FTD_rad',
    'nph_2': 'NPH_rad',
    'IncExc_nAD': 'IncExc_nAD_rad',
    'Prod_nAD': 'Prod_nAD_rad',
    'Max_nAD': 'Max_nAD_rad'
}, inplace=True)

# assert to check if the product combination worked
assert (1-cleaned_data['Prod_nAD_rad']).equals((1-cleaned_data['VD_rad']) * (1-cleaned_data['PRD_rad']) * (1-cleaned_data['FTD_rad']))
cleaned_data.head()
# %%
cleaned_data = cleaned_data.merge(model_probabilities, on='pt', how='left')
assert len(cleaned_data) == 490

# compute the combined probabilities of model + radiologist
for etiology in etiologies:
    if etiology == 'nAD':
        etiology_name = method_of_nAD
    else: 
        etiology_name = etiology
    rad_1 = cleaned_data[etiology_name + '_rad']
    model_1 = cleaned_data[etiology + '_model']
    # rad_0 = 1 - rad_1
    # model_0 = 1 - model_1
    # norm = (model_0 * rad_0) + (model_1 * rad_1) 
    # cleaned_data[etiology + '_combined'] = rad_1 * model_1 / norm
    cleaned_data[etiology + '_combined'] = (rad_1 + model_1)/2

# %%
# %% # calculate auroc and aupr for each radiologist
from sklearn.metrics import roc_auc_score, average_precision_score



# Define the true labels for the cases
true_labels = diagnosis_cases.set_index('pt')[['AD_lb', 'LBD_lb', 'VD_lb', 'PRD_lb', 'FTD_lb', 'NPH_lb']]
true_labels['nAD_lb'] = true_labels['VD_lb'] | true_labels['PRD_lb'] | true_labels['FTD_lb'] #| true_labels['NPH_lb'] 

radiologist_columns = ['doctor', 'pt'] + [
    etiology + '_rad' if etiology != 'nAD' else method_of_nAD + '_rad' 
    for etiology in etiologies
]
print(radiologist_columns)
# Calculate the AUROC and AUPR for each radiologist and each etiology

# , 'LBD', 'VD', 'PRD', 'FTD', 'NPH']
performance = {
    'AUROC': {etiology: {} for etiology in etiologies},
    'AUPR': {etiology: {} for etiology in etiologies}
}
model_performance = {
    'AUROC': {etiology: None for etiology in etiologies},
    'AUPR': {etiology: None for etiology in etiologies}
}
combined_performance = {
    'AUROC': {etiology: {} for etiology in etiologies},
    'AUPR': {etiology: {} for etiology in etiologies}
}
for etiology in etiologies:
    for radiologist in cleaned_data['doctor'].unique():
        radiologist_data = cleaned_data[cleaned_data['doctor'] == radiologist][radiologist_columns]
        # radiologist_data = radiologist_data.dropna(subset=['alz_ds', 'prod_nAD'])
        y_true = true_labels.loc[radiologist_data['pt']][etiology + '_lb']
        if etiology == 'nAD':
            y_score = radiologist_data[method_of_nAD + '_rad']
        else:
            y_score = radiologist_data[etiology + '_rad']
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        print(f"etiology: {etiology}")
        print(f"Radiologist: {radiologist}")
        print(f"AUROC: {auroc}, AUPR: {aupr}")
        performance['AUROC'][etiology][radiologist] = auroc
        performance['AUPR'][etiology][radiologist] = aupr

        # now compute combined performance
        # y_true is the same
        y_score = cleaned_data.loc[radiologist_data.index][etiology + '_combined']
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        print(f"etiology: {etiology}")
        print(f"Radiologist: {radiologist}")
        print(f"Combined")
        print(f"AUROC: {auroc}, AUPR: {aupr}")
        combined_performance['AUROC'][etiology][radiologist] = auroc
        combined_performance['AUPR'][etiology][radiologist] = aupr

    # model performance
    y_true = true_labels[etiology + '_lb']
    y_score = model_probabilities[etiology + '_model']
    # sort them both by index
    y_true = y_true.sort_index()
    y_score = y_score.sort_index()
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    print(f"etiology: {etiology}")
    print(f"Model")
    print(f"AUROC: {auroc}, AUPR: {aupr}")
    model_performance['AUROC'][etiology] = auroc
    model_performance['AUPR'][etiology] = aupr

#%% inter-rater agreement

# pairwise spearman correlation
from scipy.stats import spearmanr

rad_AD_corr = {}
rad_nAD_corr = {}
combined_AD_corr = {}
combined_nAD_corr = {}

for radiologist1 in cleaned_data['doctor'].unique():
    for radiologist2 in cleaned_data['doctor'].unique():
        if radiologist1 != radiologist2:
            rad1_data = cleaned_data[cleaned_data['doctor'] == radiologist1]
            rad2_data = cleaned_data[cleaned_data['doctor'] == radiologist2]
            # sort both by 'pt'
            rad1_data = rad1_data.sort_values(by='pt')
            rad2_data = rad2_data.sort_values(by='pt')

            rad1_AD = rad1_data['AD_rad']
            rad2_AD = rad2_data['AD_rad']
            rad1_nAD = rad1_data[method_of_nAD + '_rad']
            rad2_nAD = rad2_data[method_of_nAD + '_rad']
            rad_AD_corr[(radiologist1, radiologist2)] = spearmanr(rad1_AD, rad2_AD).correlation
            rad_nAD_corr[(radiologist1, radiologist2)] = spearmanr(rad1_nAD, rad2_nAD).correlation

            combined1_AD = rad1_data['AD_combined']
            combined2_AD = rad2_data['AD_combined']
            combined1_nAD = rad1_data['nAD_combined']
            combined2_nAD = rad2_data['nAD_combined']
            combined_AD_corr[(radiologist1, radiologist2)] = spearmanr(combined1_AD, combined2_AD).correlation
            combined_nAD_corr[(radiologist1, radiologist2)] = spearmanr(combined1_nAD, combined2_nAD).correlation



# bootstrap confidence intervals
from sklearn.utils import resample

def bootstrap_mean_ci_correlations(corr_dict, unique_doctors, n_iter=100000):
    m = len(unique_doctors)
    corr_star = []
    for _ in range(n_iter):
        # resample m doctors
        resampled_doctors = resample(unique_doctors, replace=True, n_samples=m)
        # resampled_doctors = set(resampled_doctors)
        # compute the mean correlation of the pairs
        # print(resampled_doctors)
        # for each pair , get the corr

        for i in range(m):
            for j in range(i+1, m):
                if (resampled_doctors[i], resampled_doctors[j]) in corr_dict:
                    corr_star.append(corr_dict[(resampled_doctors[i], resampled_doctors[j])])
                elif (resampled_doctors[j], resampled_doctors[i]) in corr_dict:
                    corr_star.append(corr_dict[(resampled_doctors[j], resampled_doctors[i])])
                elif resampled_doctors[i] == resampled_doctors[j]:
                    # skip
                    continue
                else:
                    raise ValueError(f"Pair not found: {resampled_doctors[i]}, {resampled_doctors[j]}")

    return np.mean(corr_star), np.percentile(corr_star, [2.5, 97.5])

    # bootstrap_correlations = np.array(bootstrap_correlations)
    # lower, upper = np.percentile(bootstrap_correlations, [2.5, 97.5])
    # return np.mean(correlations), lower, upper

print(f"AD radiologists: {bootstrap_mean_ci_correlations(rad_AD_corr, cleaned_data['doctor'].unique())}")
print(f"nAD radiologists: {bootstrap_mean_ci_correlations(rad_nAD_corr, cleaned_data['doctor'].unique())}")

print(f"AD combined: {bootstrap_mean_ci_correlations(combined_AD_corr, cleaned_data['doctor'].unique())}")
print(f"nAD combined: {bootstrap_mean_ci_correlations(combined_nAD_corr, cleaned_data['doctor'].unique())}")

# %% # plot performance as a catplot with boxplot and stripplot
# plot performance as a catplot with boxplot and stripplot
import seaborn as sns
import matplotlib.pyplot as plt

performance_df_list = []  # Create a list to collect DataFrames

for metric in performance:
    for etiology in performance[metric]:
        for radiologist in performance[metric][etiology]:
            df = pd.DataFrame({
                'Metric': [metric],  # Wrap single values in a list
                'Score': [performance[metric][etiology][radiologist]],  # Wrap single values in a list
                'Etiology': [etiology],  # Wrap single values in a list
                'Radiologist': [radiologist],  # Wrap single values in a list
                'Method': ['Radiologist']
            })
            performance_df_list.append(df)  # Append each DataFrame to the list

# Add the model performance to the list
for metric in model_performance:
    for etiology in model_performance[metric]:
        df = pd.DataFrame({
            'Metric': [metric],  # Wrap single values in a list
            'Score': [model_performance[metric][etiology]],  # Wrap single values in a list
            'Etiology': [etiology],  # Wrap single values in a list
            'Radiologist': ['Model'],  # Wrap single values in a list
            'Method': ['Model']
        })
        performance_df_list.append(df)  # Append each DataFrame to the list

# Add the combined performance to the list
for metric in combined_performance:
    for etiology in combined_performance[metric]:
        for radiologist in combined_performance[metric][etiology]:
            df = pd.DataFrame({
                'Metric': [metric],  # Wrap single values in a list
                'Score': [combined_performance[metric][etiology][radiologist]],  # Wrap single values in a list
                'Etiology': [etiology],  # Wrap single values in a list
                'Radiologist': [radiologist],  # Wrap single values in a list
                'Method': ['Combined']
            })
            performance_df_list.append(df)  # Append each DataFrame to the list
# Concatenate all DataFrames in the list into a single DataFrame
performance_df = pd.concat(performance_df_list, ignore_index=True)

print(performance_df.head())
# %% #plot
method_order = ['Radiologist', 'Combined']
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})



#AUROC filtered and sorted df 
AUROC_AD_raw = performance_df[(performance_df['Metric'] == 'AUROC') & (performance_df['Etiology'] == 'AD') & (performance_df['Method'] != 'Model')]
AUROC_AD_sorted = AUROC_AD_raw.sort_values(by=['Radiologist', 'Method'])

radiologists = AUROC_AD_sorted['Radiologist'].unique()

palette = {radiologist: 'orange' for radiologist in radiologists}

def wilcoxon_signed_rank(df):
    pivot_df = df.pivot(index='Radiologist', columns='Method', values='Score')

    # Extract the paired scores
    combined_scores = pivot_df['Combined']
    radiologist_scores = pivot_df['Radiologist']

    # Perform the one-sided Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(combined_scores, radiologist_scores, alternative='greater')

    # Output the results
    print(f'Wilcoxon signed-rank test statistic: {statistic}')
    print(f'p-value: {p_value}')

    return statistic, p_value

AUROC_nAD_raw = performance_df[(performance_df['Metric'] == 'AUROC') & (performance_df['Etiology'] == 'nAD') & (performance_df['Method'] != 'Model')]
AUROC_nAD_sorted = AUROC_nAD_raw.sort_values(by=['Radiologist', 'Method'])

#pull model value
model_value_AD = performance_df[(performance_df['Metric'] == 'AUROC') & (performance_df['Etiology'] == 'AD') & (performance_df['Method'] == 'Model')]['Score'].mean()
model_value_nAD = performance_df[(performance_df['Metric'] == 'AUROC') & (performance_df['Etiology'] == 'nAD') & (performance_df['Method'] == 'Model')]['Score'].mean()
print(performance_df[(performance_df['Metric'] == 'AUROC') & (performance_df['Etiology'] == 'AD') & (performance_df['Method'] == 'Model')]['Score'])
#shared axis limits
y_min = min(AUROC_AD_sorted['Score'].min(), AUROC_nAD_sorted['Score'].min())-0.15
y_max = max(AUROC_AD_sorted['Score'].max(), AUROC_nAD_sorted['Score'].max())+0.15

#AUROC plot
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

#AUROC AD plot
sns.boxplot(x='Method', y='Score', data=AUROC_AD_sorted,order=method_order, ax=axes[0])
sns.swarmplot(x='Method', y='Score', data=AUROC_AD_sorted,order=method_order, size=6, ax=axes[0])
sns.lineplot(x='Method', y='Score', hue='Radiologist',palette=palette, data=AUROC_AD_sorted, linewidth=1, ax=axes[0])
axes[0].axhline(y=model_value_AD, linestyle='--', color='green', label='Model Median', linewidth = 1)
axes[0].text(-0.40, model_value_AD + 0.03, 'Independent Model', va='center', ha='left', color='green', fontsize=8)
axes[0].set_title('AD Prediction', fontsize=10)
axes[0].set_xlabel('', fontsize=10)
axes[0].set_ylabel('AUROC Score', fontsize=10)
axes[0].set_xticklabels(['Independent\nRadiologist', 'Radiologist \n+ AI Model'])
axes[0].tick_params(axis='both', which='major', labelsize=10)
axes[0].legend_.remove()  # Hide the legend

#AUROC nAD plot
sns.boxplot(x='Method', y='Score', data=AUROC_nAD_sorted,order=method_order, ax=axes[1])
sns.swarmplot(x='Method', y='Score', data=AUROC_nAD_sorted,order=method_order, size = 6, ax=axes[1])
sns.lineplot(x='Method', y='Score', hue='Radiologist',palette=palette, data=AUROC_nAD_sorted, linewidth = 1, ax=axes[1], alpha = 0.7)
axes[1].axhline(y=model_value_nAD, linestyle='--', color='green', label='Model Median', linewidth = 1)
axes[1].text(-0.40, model_value_nAD + 0.03, 'Independent Model', va='center', ha='left', color='green', fontsize=8)
axes[1].set_title('OIED Prediction', fontsize=10)
axes[1].set_xlabel('', fontsize=10)
axes[1].set_ylabel('AUROC Score', fontsize=10)
axes[1].set_xticklabels(['Independent\nRadiologist', 'Radiologist \n+ AI Model'])
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].legend_.remove()  # Hide the legend

# Apply common y-axis limits
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

# Adjust layout
plt.tight_layout()
# plt.savefig('/projectnb/vkolagrp/spuduch/plots/test_AUROC.svg')
plt.show()
plt.close()

#AUPR filtered and sort
AUPR_AD_raw = performance_df[(performance_df['Metric'] == 'AUPR') & (performance_df['Etiology'] == 'AD') & (performance_df['Method'] != 'Model')]
AUPR_AD_sorted = AUPR_AD_raw.sort_values(by=['Radiologist', 'Method'])
# print(AUPR_AD_sorted)
AUPR_nAD_raw = performance_df[(performance_df['Metric'] == 'AUPR') & (performance_df['Etiology'] == 'nAD') & (performance_df['Method'] != 'Model')]
AUPR_nAD_sorted = AUPR_nAD_raw.sort_values(by=['Radiologist', 'Method'])

#aupr shared axis limits 
y_min = min(AUPR_AD_sorted['Score'].min(), AUPR_nAD_sorted['Score'].min())-0.15
y_max = max(AUPR_AD_sorted['Score'].max(), AUPR_nAD_sorted['Score'].max())+0.15

#pull model value
model_value_AD = performance_df[(performance_df['Metric'] == 'AUPR') & (performance_df['Etiology'] == 'AD') & (performance_df['Method'] == 'Model')]['Score'].mean()
model_value_nAD = performance_df[(performance_df['Metric'] == 'AUPR') & (performance_df['Etiology'] == 'nAD') & (performance_df['Method'] == 'Model')]['Score'].mean()

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

#AUPR AD
sns.boxplot(x='Method', y='Score', data=AUPR_AD_sorted,order=method_order, ax=axes[0])
sns.swarmplot(x='Method', y='Score', data=AUPR_AD_sorted,order=method_order, size=6, ax=axes[0])
sns.lineplot(x='Method', y='Score', hue='Radiologist',palette=palette, data=AUPR_AD_sorted, linewidth = 1, ax=axes[0], alpha = 0.7)
axes[0].axhline(y=model_value_AD, linestyle='--', color='green', label='Model Median', linewidth = 1)
axes[0].text(-0.40, model_value_AD + 0.03, 'Independent Model', va='center', ha='left', color='green', fontsize=8)
axes[0].set_title('AD Prediction', fontsize=10)
axes[0].set_xlabel('', fontsize=10)
axes[0].set_ylabel('AUPR Score', fontsize=10)
axes[0].set_xticklabels(['Independent\nRadiologist', 'Radiologist \n+ AI Model'])
axes[0].tick_params(axis='both', which='major', labelsize=10)
axes[0].legend_.remove()  # Hide the legend

#AUPR nAD
sns.boxplot(x='Method', y='Score', data=AUPR_nAD_sorted,order=method_order, ax=axes[1])
sns.swarmplot(x='Method', y='Score', data=AUPR_nAD_sorted,order=method_order, size=6, ax=axes[1])
sns.lineplot(x='Method', y='Score', hue='Radiologist',palette=palette, data=AUPR_nAD_sorted, linewidth = 1, ax=axes[1], alpha = 0.7)
axes[1].axhline(y=model_value_nAD, linestyle='--', color='green', label='Model Median', linewidth = 1)
axes[1].text(-0.40, model_value_nAD + 0.03, 'Independent Model', va='center', ha='left', color='green', fontsize=8)
axes[1].set_title('OIED Prediction', fontsize=10)
axes[1].set_xlabel('', fontsize=10)
axes[1].set_ylabel('AUPR Score', fontsize=10)
axes[1].set_xticklabels(['Independent\nRadiologist', 'Radiologist \n+ AI Model'])
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].legend_.remove()  # Hide the legend

# Apply common y-axis limits
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

# Adjust layout
plt.tight_layout()
# plt.savefig('/projectnb/vkolagrp/spuduch/plots/test_AUPR.svg')
plt.show()
plt.close()
# %% stats
# Perform the Wilcoxon signed-rank test for AUROC
print('AD AUROC')
wilcoxon_signed_rank(AUROC_AD_raw)
print('nAD AUROC')
wilcoxon_signed_rank(AUROC_nAD_raw)

# Perform the Wilcoxon signed-rank test for AUPR
print('AD AUPR')
wilcoxon_signed_rank(AUPR_AD_raw)
print('nAD AUPR')
wilcoxon_signed_rank(AUPR_nAD_raw)

# %% percent increase in mean

AD_AUROC_radiologist = AUROC_AD_raw[AUROC_AD_raw['Method'] == 'Radiologist']['Score'].mean()
AD_AUROC_combined = AUROC_AD_raw[AUROC_AD_raw['Method'] == 'Combined']['Score'].mean()

nAD_AUROC_radiologist = AUROC_nAD_raw[AUROC_nAD_raw['Method'] == 'Radiologist']['Score'].mean()
nAD_AUROC_combined = AUROC_nAD_raw[AUROC_nAD_raw['Method'] == 'Combined']['Score'].mean()

AD_AUPR_radiologist = AUPR_AD_raw[AUPR_AD_raw['Method'] == 'Radiologist']['Score'].mean()
AD_AUPR_combined = AUPR_AD_raw[AUPR_AD_raw['Method'] == 'Combined']['Score'].mean()

nAD_AUPR_radiologist = AUPR_nAD_raw[AUPR_nAD_raw['Method'] == 'Radiologist']['Score'].mean()
nAD_AUPR_combined = AUPR_nAD_raw[AUPR_nAD_raw['Method'] == 'Combined']['Score'].mean()

print(f'AD AUROC percent increase: {(AD_AUROC_combined - AD_AUROC_radiologist) / AD_AUROC_radiologist * 100:.2f}%')
print(f'nAD AUROC percent increase: {(nAD_AUROC_combined - nAD_AUROC_radiologist) / nAD_AUROC_radiologist * 100:.2f}%')

print(f'AD AUPR percent increase: {(AD_AUPR_combined - AD_AUPR_radiologist) / AD_AUPR_radiologist * 100:.2f}%')
print(f'nAD AUPR percent increase: {(nAD_AUPR_combined - nAD_AUPR_radiologist) / nAD_AUPR_radiologist * 100:.2f}%')
# %% just the radiologist, not combined
import matplotlib.lines as mlines

# Filter and prepare AUROC data for radiologists
AUROC_filtered = performance_df[
    (performance_df['Metric'] == 'AUROC') &
    (performance_df['Method'] == 'Radiologist')
].copy()
AUROC_filtered['Etiology_Label'] = AUROC_filtered['Etiology'].map({'AD': 'AD', 'nAD': 'OIED'})

# Get model scores for AUROC
model_scores = performance_df[
    (performance_df['Metric'] == 'AUROC') & 
    (performance_df['Method'] == 'Model')
].copy()
model_scores['Etiology_Label'] = model_scores['Etiology'].map({'AD': 'AD', 'nAD': 'OIED'})

plt.figure(figsize=(4, 3))
sns.boxplot(x='Etiology_Label', y='Score', data=AUROC_filtered, color='dodgerblue')

# Plot radiologist scores in orange
sns.swarmplot(x='Etiology_Label', y='Score', data=AUROC_filtered, color='orange', size=6)

# Plot model scores as a green diamond point
sns.swarmplot(x='Etiology_Label', y='Score', data=model_scores, color='green', size=8, marker='D')

# Create custom legend handles
radiologist_handle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=6, label='Radiologist')
model_handle = mlines.Line2D([], [], color='green', marker='D', linestyle='None', markersize=8, label='Model')
plt.legend(handles=[radiologist_handle, model_handle], loc='lower right')

plt.ylabel('AUROC Score')
plt.xlabel('')
plt.ylim(AUROC_filtered['Score'].min() - 0.15, AUROC_filtered['Score'].max() + 0.15)
plt.tight_layout()
plt.show()
plt.savefig('/projectnb/vkolagrp/spuduch/plots/test_AUROC_radiologist.svg')
plt.close()

import matplotlib.lines as mlines

# Filter and prepare AUPR data for radiologists
AUPR_filtered = performance_df[
    (performance_df['Metric'] == 'AUPR') &
    (performance_df['Method'] == 'Radiologist')
].copy()
AUPR_filtered['Etiology_Label'] = AUPR_filtered['Etiology'].map({'AD': 'AD', 'nAD': 'OIED'})

# Get model scores for AUPR
model_scores = performance_df[
    (performance_df['Metric'] == 'AUPR') & 
    (performance_df['Method'] == 'Model')
].copy()
model_scores['Etiology_Label'] = model_scores['Etiology'].map({'AD': 'AD', 'nAD': 'OIED'})

plt.figure(figsize=(4, 3))
sns.boxplot(x='Etiology_Label', y='Score', data=AUPR_filtered, color='dodgerblue')

# Plot radiologist scores in orange
sns.swarmplot(x='Etiology_Label', y='Score', data=AUPR_filtered, color='orange', size=6)

# Plot model scores as a green diamond point
sns.swarmplot(x='Etiology_Label', y='Score', data=model_scores, color='green', size=8, marker='D')

# Create custom legend handles
radiologist_handle = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=6, label='Radiologist')
model_handle = mlines.Line2D([], [], color='green', marker='D', linestyle='None', markersize=8, label='Model')
plt.legend(handles=[radiologist_handle, model_handle])

plt.ylabel('AUPR Score')
plt.xlabel('')
plt.ylim(AUPR_filtered['Score'].min() - 0.15, AUPR_filtered['Score'].max() + 0.15)
plt.tight_layout()
plt.show()
plt.savefig('/projectnb/vkolagrp/spuduch/plots/test_AUPR_radiologist.svg')
plt.close()
# %% agreement

cleaned_data
# %% # plot roc curve for each radiologist and also a roc curve for model
from sklearn.metrics import roc_curve, precision_recall_curve

# AD first
roc_fig, roc_ax = plt.subplots(figsize=(4, 4))
pr_fig, pr_ax = plt.subplots(figsize=(4, 4))

radiologist_aurocs = []
radiologist_auprs = []
for radiologist in cleaned_data['doctor'].unique():
    radiologist_data = cleaned_data[cleaned_data['doctor'] == radiologist][radiologist_columns]
    # print(radiologist_data)
    # radiologist_data = radiologist_data.dropna(subset=['alz_ds', 'prod_nAD'])
    y_true = true_labels.loc[radiologist_data['pt']]['AD_lb']
    # print(y_true)
    y_score = radiologist_data['AD_rad']
    y_true = y_true.sort_index()
    y_score = y_score.sort_index()

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    radiologist_aurocs.append(auroc)
    radiologist_auprs.append(aupr)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_ax.plot(fpr, tpr, alpha=0.5, color='blue', label = 'radiologist', linewidth=0.75)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_ax.plot(recall, precision, alpha=0.5, color='blue', label = 'radiologist', linewidth=0.75)

mean_auroc, std_auroc = np.mean(radiologist_aurocs), np.std(radiologist_aurocs)
mean_aupr, std_aupr = np.mean(radiologist_auprs), np.std(radiologist_auprs)
# draw the model roc curve
y_true = true_labels['AD_lb']
y_score = model_probabilities['AD_model']
y_true = y_true.sort_index()
y_score = y_score.sort_index()

auroc = roc_auc_score(y_true, y_score)
aupr = average_precision_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_ax.plot(fpr, tpr, label=f"Model AUROC: {auroc:.2f}", color='green', linewidth=1.5)

roc_ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')

#fix legend so that they aren't repeated
handles, labels = roc_ax.get_legend_handles_labels()
roc_legend = dict(zip(labels, handles))
roc_legend[f'Radiologist \nMean AUROC: {mean_auroc:.2f} ± {std_auroc:.2f}'] = roc_legend.pop('radiologist')
roc_ax.legend(roc_legend.values(), roc_legend.keys(), loc='lower right')

precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_ax.plot(recall, precision, label=f"Model AUPR: {aupr:.2f}", color='green', linewidth=1.5)
pr_ax.set_xlabel('Recall')
pr_ax.set_ylabel('Precision')

# fix legend
handles, labels = pr_ax.get_legend_handles_labels()
pr_legend = dict(zip(labels, handles))
pr_legend[f'Radiologist \nMean AUPR: {mean_aupr:.2f} ± {std_aupr:.2f}'] = pr_legend.pop('radiologist')
pr_ax.legend(pr_legend.values(), pr_legend.keys(), loc='lower right')


roc_ax.set_title('AD ROC Curve')
pr_ax.set_title('AD PR Curve')

#save 
# roc_fig.savefig('/projectnb/vkolagrp/spuduch/plots/test_AD_roc.svg')
# pr_fig.savefig('/projectnb/vkolagrp/spuduch/plots/test_AD_pr.svg')

plt.close()

# nAD
roc_fig, roc_ax = plt.subplots(figsize=(4, 4))
pr_fig, pr_ax = plt.subplots(figsize=(4, 4))

radiologist_aurocs = []
radiologist_auprs = []
for radiologist in cleaned_data['doctor'].unique():
    radiologist_data = cleaned_data[cleaned_data['doctor'] == radiologist][radiologist_columns]
    # radiologist_data = radiologist_data.dropna(subset=['alz_ds', 'prod_nAD'])
    y_true = true_labels.loc[radiologist_data['pt']]['nAD_lb']
    y_score = radiologist_data[method_of_nAD + '_rad']
    y_true = y_true.sort_index()
    y_score = y_score.sort_index()

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    radiologist_aurocs.append(auroc)
    radiologist_auprs.append(aupr)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_ax.plot(fpr, tpr, alpha=0.5, color='blue', label = 'radiologist', linewidth=0.75)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_ax.plot(recall, precision, alpha=0.5, color='blue', label = 'radiologist', linewidth=0.75)

mean_auroc, std_auroc = np.mean(radiologist_aurocs), np.std(radiologist_aurocs)
mean_aupr, std_aupr = np.mean(radiologist_auprs), np.std(radiologist_auprs)

# draw the model roc curve
y_true = true_labels['nAD_lb']
y_score = model_probabilities['nAD' + '_model']
y_true = y_true.sort_index()
y_score = y_score.sort_index()

auroc = roc_auc_score(y_true, y_score)
aupr = average_precision_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_ax.plot(fpr, tpr, label=f"Model AUROC: {auroc:.2f}", color='green', linewidth=1.5)

roc_ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')

#fix legend so that they aren't repeated
handles, labels = roc_ax.get_legend_handles_labels()
roc_legend = dict(zip(labels, handles))
roc_legend[f'Radiologist \nMean AUROC: {mean_auroc:.2f} ± {std_auroc:.2f}'] = roc_legend.pop('radiologist')
roc_ax.legend(roc_legend.values(), roc_legend.keys(), loc='lower right')

precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_ax.plot(recall, precision, label=f"Model AUPR: {aupr:.2f}", color='green', linewidth=1.5)
pr_ax.set_xlabel('Recall')
pr_ax.set_ylabel('Precision')

# fix legend
handles, labels = pr_ax.get_legend_handles_labels()
pr_legend = dict(zip(labels, handles))
pr_legend[f'Radiologist \nMean AUPR: {mean_aupr:.2f} ± {std_aupr:.2f}'] = pr_legend.pop('radiologist')
pr_ax.legend(pr_legend.values(), pr_legend.keys(), loc='lower right')


roc_ax.set_title('OIED ROC Curve')
pr_ax.set_title('OIED PR Curve')

#save
# roc_fig.savefig('/projectnb/vkolagrp/spuduch/plots/test_nAD_roc.svg')
# pr_fig.savefig('/projectnb/vkolagrp/spuduch/plots/test_nAD_pr.svg')




# %%

# plot AD_pred vs nAD_pred from the model_probabilities

scatter_data = cleaned_data.copy()
scatter_data = scatter_data.drop_duplicates(subset='CASEID')
scatter_data
# plt.scatter(cleaned_data['AD_model'], cleaned_data['nAD_model'], c=scatter_data['doctor'], cmap='viridis')