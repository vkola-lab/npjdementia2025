import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    # Define paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rad_path = os.path.join(repo_root, 'data', 'radiologist_data', 'cleaned_radiologist_data.csv')
    cases_path = os.path.join(repo_root, 'data', 'radiologist_data', 'clinician_review_cases_converted_radio.csv')
    model_path = os.path.join(repo_root, 'data', 'results_data', 'ml_test_data_AD_nAD(FTD,VD,PRD)_pred.csv')
    output_dir = os.path.join(repo_root, 'figs')

    # save flag
    save_figs = True
    show_legend = True  # Toggle to show/hide legend

    individual_lineplots = False  # Toggle to show individual lineplots for each radiologist
    agg_boxplots = False  # Toggle to show boxplot aggregation at sampled weights
    agg_lineplot = True  # Toggle aggregated mean ±1 STD lineplot

    os.makedirs(output_dir, exist_ok=True)

    # Load radiologist ratings
    rad_df = pd.read_csv(rad_path)
    #compute radiologist nAD label as max of  'vasdem', 'prion_d', 'ftld_v'
    # AD label is just alz_ds
    rad_df['nAD_rad'] = rad_df[['vasdem', 'prion_d', 'ftld_v']].max(axis=1)
    rad_df['nAD_rad'] = rad_df['nAD_rad'] / 100
    rad_df['AD_rad'] = rad_df['alz_ds'] / 100
    # Load true labels
    cases = pd.read_csv(cases_path)
    cases['pt'] = cases['case_number'].str.replace('CASE_', '').str.strip().astype(int)
    # Compute nAD label as union of VD, PRD, FTD
    cases['nAD_lb'] = cases[['VD_lb', 'PRD_lb', 'FTD_lb']].any(axis=1).astype(int)
    true_labels = cases.set_index('pt')[['AD_lb', 'nAD_lb']]

    # Load model predictions
    model_df = pd.read_csv(model_path)
    model_df = model_df.merge(cases[['ID', 'pt']], left_on='NACCID', right_on='ID', how='left')
    model_df.set_index('pt', inplace=True)
    model_df.rename(columns={'AD': 'AD_model', 'nAD': 'nAD_model'}, inplace=True)

    # Merge radiologist and model
    data = rad_df.merge(model_df[['AD_model', 'nAD_model']], on='pt', how='left')

    doctors = data['doctor'].unique()
    weights = np.linspace(0, 1, 11)
    metrics = {'AUROC': roc_auc_score, 'AUPR': average_precision_score}
    etis = ['AD', 'nAD']
    # Radiologist columns mapping for nAD
    rad_col_map = {'AD': 'AD_rad', 'nAD': 'nAD_rad'}

    for met_name, met_fun in metrics.items():
        title_map = {'AD': 'AD', 'nAD': 'OIED'}
        if agg_lineplot:
            color_map = {'AD': 'turquoise',
                         'nAD': 'coral'}
            label_map = {'AD': 'AD', 'nAD': 'OIED'}
            linestyle_map = {'AD': '-', 'nAD': '--'}
            means_dict, ses_dict = {}, {}
            n_docs = len(doctors)
            plt.figure(figsize=(5,4))
            interval_idx = [0, 2, 4, 6, 8, 10]  # weights 0, 0.2, ..., 1.0
            for eti in etis:
                means, ses = [], []
                for w in weights:
                    scores = []
                    for doc in doctors:
                        sub = data[data['doctor'] == doc]
                        eti_lb = f'{eti}_lb'
                        y_true = true_labels.loc[sub['pt'], eti_lb]
                        y_pred = (1 - w) * sub[rad_col_map[eti]] + w * sub[f'{eti}_model']
                        scores.append(met_fun(y_true, y_pred))
                    means.append(np.mean(scores))
                    ses.append(np.std(scores) / np.sqrt(n_docs))
                means_dict[eti] = means
                ses_dict[eti] = ses
                # plt.plot(weights, means, color=color_map[eti], linestyle=linestyle_map[eti], label=f"{label_map[eti]} Mean")
                plt.errorbar(np.array(weights)[interval_idx], np.array(means)[interval_idx],
                             yerr=np.array(ses)[interval_idx], fmt='o', color=color_map[eti], capsize=5, linestyle=linestyle_map[eti], label=f"{label_map[eti]} ±1 SE")

            plt.xlabel('Model Weight (%)', fontsize=12)
            plt.ylabel(f'{met_name} of individual radiologists + AI', fontsize=12)
            plt.xticks(ticks=np.array(weights)[interval_idx], labels=[f'{int(w*100)}%' for w in np.array(weights)[interval_idx]])
            # plt.title(f"AD & OIED {met_name} Mean±SE vs Model Weight")
            if show_legend:
                plt.legend(title=None)
            plt.tight_layout()
            fname_line = f'aggline_ADnAD_{met_name}.svg'
            if save_figs:
                plt.savefig(os.path.join(output_dir, fname_line))
                print(f"Saved combined lineplot to: {os.path.join(output_dir, fname_line)}")
            else:
                plt.show()
            plt.close()
            continue  # skip the rest of the loop for agg_lineplot
        for eti in etis:
            if individual_lineplots:
                plt.figure(figsize=(5,4))
                for doc in doctors:
                    sub = data[data['doctor'] == doc]
                    eti_lb = f'{eti}_lb'
                    y_true = true_labels.loc[sub['pt'], eti_lb]
                    scores = []
                    for w in weights:
                        y_pred = (1 - w) * sub[rad_col_map[eti]] + w * sub[f'{eti}_model']
                        scores.append(met_fun(y_true, y_pred))
                    plt.plot(weights, scores, label=doc)
                plt.xlabel('Model Weight', fontsize=12)
                plt.ylabel(met_name, fontsize=12)
                plt.xticks(ticks=weights, labels=[f'{int(w*100)}%' for w in weights])
                plt.title(f"{title_map[eti]} {met_name} vs Model Weight")
                if show_legend:
                    plt.legend(title='Radiologist', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                fname = f'weighting_{eti}_{met_name}.svg'
                if save_figs:
                    plt.savefig(os.path.join(output_dir, fname))
                    print(f"Saved figure to: {os.path.join(output_dir, fname)}")
                else:
                    plt.show()
                plt.close()

            # Boxplot aggregation at sampled weights
            if agg_boxplots:
                sample_weights = np.arange(0, 1.001, 0.15)
                records = []
                for w in sample_weights:
                    for doc in doctors:
                        sub = data[data['doctor'] == doc]
                        eti_lb = f'{eti}_lb'
                        y_true = true_labels.loc[sub['pt'], eti_lb]
                        y_pred = (1 - w) * sub[rad_col_map[eti]] + w * sub[f'{eti}_model']
                        score = met_fun(y_true, y_pred)
                        records.append({'Weight': w, 'Score': score, 'Radiologist': doc})
                df_box = pd.DataFrame(records)
                plt.figure(figsize=(5,4))
                sns.boxplot(x='Weight', y='Score', data=df_box)
                sns.swarmplot(x='Weight', y='Score', data=df_box, color='orange', size=6)
                plt.xlabel('Model Weight', fontsize=12)
                plt.ylabel(f'{met_name} of individual radiologists + AI', fontsize=12)
                plt.xticks(ticks=sample_weights, labels=[f'{int(w*100)}%' for w in sample_weights])
                title_map = {'AD': 'AD', 'nAD': 'OIED'}
                plt.title(f"{title_map[eti]} {met_name} Distribution at Sampled Weights")
                if show_legend:
                    # swarms/boxes share same color legend
                    plt.legend([],[], frameon=False)
                plt.tight_layout()
                fname_box = f'box_{eti}_{met_name}.svg'
                if save_figs:
                    plt.savefig(os.path.join(output_dir, fname_box))
                    print(f"Saved boxplot to: {os.path.join(output_dir, fname_box)}")
                else:
                    plt.show()
                plt.close()

if __name__ == '__main__':
    main()