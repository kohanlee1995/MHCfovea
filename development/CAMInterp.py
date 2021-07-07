import os, sys, re, json, random, importlib
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker as lm
from venn import venn
from venn import generate_petal_labels, draw_venn
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from util import *
import warnings
warnings.filterwarnings('ignore')


class CAMInterp():
    def __init__(self, mhc_seq_filename, allele_mask_dirname, epitope_mask_dirname, df_filename, output_dir,
                 pred_basename='score', pred_threshold=0.9, mhc_len=182, min_sample_num=100, submotif_len=4):
        self.aa_str = 'ACDEFGHIKLMNPQRSTVWY'
        
        self.mhc_len = mhc_len
        self.epitope_len = 10
        self.res34_pos = [6, 8, 23, 44, 58, 61, 62, 65, 66, 68, 69, 72, 73, 75, 76, 79, 80, 83, 94,
                          96, 98, 113, 115, 117, 142, 146, 149, 151, 155, 157, 158, 162, 166, 170]
        
        self.color_dict = {'A': '#DACC47', 'B': '#B1DEC9', 'C': '#FFBB99', 'polymorphism': '#875A85'}
        
        self.dpi = 600
        self.fontsize = 10

        self.pred_basename = pred_basename
        self.pred_threshold = pred_threshold
        self.min_sample_num = min_sample_num
        self.submotif_len = submotif_len

        self.output_dir = output_dir

        # mhc_seq_dict
        self.mhc_seq_dict = json.load(open(mhc_seq_filename, 'r'))

        # allele_mask_df
        if type(allele_mask_dirname) == list:
            alleles = [self._convert_allele(i) for i in os.listdir(allele_mask_dirname[0])]
            self.allele_mask_df = pd.DataFrame(columns=alleles, index=range(self.mhc_len), data=0)
            self.allele_mask_df.loc['count'] = 0
            for i in range(len(allele_mask_dirname)):
                temp_df = pd.DataFrame(self._parse_mask(allele_mask_dirname[i], mask_type='mhc'))
                self.allele_mask_df.loc[temp_df.index, temp_df.columns] += temp_df
                self.allele_mask_df.loc['count', temp_df.columns] += 1
            self.allele_mask_df = self.allele_mask_df.loc[:, self.allele_mask_df.loc['count'] != 0]
            self.allele_mask_df.loc[range(self.mhc_len)] /= self.allele_mask_df.loc['count']
            self.allele_mask_df = self.allele_mask_df.drop('count')
        else:
            self.allele_mask_df = pd.DataFrame(self._parse_mask(allele_mask_dirname, mask_type='mhc'))
        self.allele_mask_df.to_csv('%s/AlleleMask.csv'%self.output_dir)

        # epitope_mask_df
        if type(epitope_mask_dirname) == list:
            alleles = [self._convert_allele(i) for i in os.listdir(epitope_mask_dirname[0])]
            self.epitope_mask_df = pd.DataFrame(columns=alleles, index=range(self.epitope_len), data=0)
            self.epitope_mask_df.loc['count'] = 0
            for i in range(len(epitope_mask_dirname)):
                temp_df = pd.DataFrame(self._parse_mask(epitope_mask_dirname[i], mask_type='epitope'))
                self.epitope_mask_df.loc[temp_df.index, temp_df.columns] += temp_df
                self.epitope_mask_df.loc['count', temp_df.columns] += 1
            self.epitope_mask_df = self.epitope_mask_df.loc[:, self.epitope_mask_df.loc['count'] != 0]
            self.epitope_mask_df.loc[range(self.epitope_len)] /= self.epitope_mask_df.loc['count']
            self.epitope_mask_df = self.epitope_mask_df.drop('count')
        else:
            self.epitope_mask_df = pd.DataFrame(self._parse_mask(epitope_mask_dirname, mask_type='epitope'))
        self.epitope_mask_df['position'] = [1,2,3,4,5,-5,-4,-3,-2,-1]
        self.epitope_mask_df = self.epitope_mask_df.set_index('position', drop=True)
        self.epitope_mask_df.to_csv('%s/EpitopeMask.csv'%self.output_dir)

        # df
        self.df = pd.read_csv(df_filename, index_col=0)
        self.alleles = list(self.df['mhc'].unique())
        self.allele_num = len(self.alleles)

        # motif_dict
        self.motif_dict = self._parse_motif(pred_basename, pred_threshold, self.min_sample_num)
        self.alleles = list(self.df['mhc'].unique())
        self.allele_num = len(self.alleles)

        # mhc_seqlogo_df
        self.mhc_seqlogo_df = self._mhc_seqlogo_df(self.alleles, list(range(self.mhc_len)))


    def ResidueAnalysis(self, cam_threshold, importance_threshold, barplot_figsize=(10,2), square_figsize=(3.5,3.5)):
        # mean plot
        self._residue_barplot(self.allele_mask_df.mean(axis=1), self.res34_pos, figsize=barplot_figsize,
                              figfile='%s/CAMmean.png'%self.output_dir)

        # importance plot
        importance_count = self._residue_importance_count(self.alleles, cam_threshold)
        self._residue_barplot(importance_count, self.res34_pos, figsize=barplot_figsize,
                              figfile='%s/CAMimportance.png'%self.output_dir)
    
        # important residues - stacked plot
        df = self._importance_stacked_barplot(cam_threshold, self.res34_pos,
                                              xticklabels=False, yticklabels=True, figsize=barplot_figsize,
                                              figfile='%s/CAMimportanceStacked.png'%self.output_dir)
        df.to_csv('%s/ImportanceStack.csv'%self.output_dir)
        
        # important residues
        residue_dict = self._select_residue(cam_threshold, importance_threshold)
        json.dump(residue_dict, open('%s/ResidueSelection.json'%self.output_dir, 'w'))
        
        # venn diagram of residue selection
        self._importance_venn_plot(residue_dict, figsize=square_figsize,
                                   figfile='%s/ResidueSelectionVenn.png'%self.output_dir)

        # correlation between residue importance and sequence entropy
        # entropy = sigma(probability**2)
        # allele part
        df = self._mhc_importance_polymorphism_plot(cam_threshold, residue_dict, figsize=square_figsize,
                                                    figfile='%s/AlleleImportanceEntropyCorrelation.png'%self.output_dir)
        df.to_csv('%s/AlleleImportancePolymorphism.csv'%self.output_dir)

        # epitope part
        df = self._epitope_importance_polymorphism_plot(figsize=square_figsize,
                                                        figfile='%s/EpitopeImportanceEntropyCorrelation.png'%self.output_dir)
        df.to_csv('%s/EpitopeImportancePolymorphism.csv'%self.output_dir)


    def ClusterAnalysis(self, method, metric, allele_figsize=(10,2), epitope_figsize=(3.5,3.5)):
        alleles = self.alleles

        # allele masks
        allele_order, position_order = self._mask_clustering_plot(alleles, mask_type='mhc',
                                                                  method=method, metric=metric,
                                                                  xticklabels=False, yticklabels=False,
                                                                  row_colors=True, figsize=allele_figsize,
                                                                  title=None, xlabel='MHC-I position', ylabel='MHC-I allele',
                                                                  figfile='%s/AlleleCAMcluster_all.png'%self.output_dir)
        
        # epitope masks
        allele_order, position_order = self._mask_clustering_plot(alleles, mask_type='epitope',
                                                                  method=method, metric=metric,
                                                                  xticklabels=True, yticklabels=False,
                                                                  row_colors=True, figsize=epitope_figsize,
                                                                  title=None, xlabel='peptide position', ylabel='MHC-I allele',
                                                                  figfile='%s/EpitopeCAMcluster_all.png'%self.output_dir)


    """"""""""""""""""""""""""""""""""""""
    # Plots
    """"""""""""""""""""""""""""""""""""""
    # mask_type: mhc or epitope
    def _mask_clustering_plot(self, alleles, mask_type='mhc',
                              method='average', metric='euclidean',
                              allele_linkage=True, position_linkage=False,
                              row_colors=False, xticklabels=True, yticklabels=True,
                              title=None, xlabel=None, ylabel=None,
                              figsize=(8, 4), figfile=None):
        # residue positions
        if mask_type == 'mhc':
            positions = list(range(self.mhc_len))
            df = self.allele_mask_df.iloc[positions][alleles].T
        else:
            positions = [1,2,3,4,-4,-3,-2,-1]
            df = self.epitope_mask_df.loc[positions][alleles].T
        
        # linkage
        zx, zy = None, None
        if allele_linkage:
            zy = hierarchy.linkage(df, method=method, metric=metric, optimal_ordering=True)
        if position_linkage:
            zx = hierarchy.linkage(df.T, method=method, metric=metric, optimal_ordering=True)
            
        # row colors
        if row_colors:
            color_list = list()
            for allele in alleles:
                hla = allele.split('*')[0]
                color_list.append(self.color_dict[hla])
        else:
            color_list = None
        
        # clustermap
        g = sns.clustermap(df,
                           col_cluster=position_linkage,
                           row_cluster=allele_linkage,
                           row_linkage=zy,
                           col_linkage=zx,
                           row_colors = color_list,
                           cmap='Blues',
                           cbar_kws={'orientation': 'horizontal', 'label': 'mask score'},
                           cbar_pos=(.3, -.05, .4, .02),
                           dendrogram_ratio=0.1,
                           colors_ratio=0.02,
                           xticklabels=xticklabels,
                           yticklabels=yticklabels,
                           figsize=figsize)
        
        g.ax_heatmap.set_title(title)
        g.ax_heatmap.set_xlabel(xlabel)
        g.ax_heatmap.set_ylabel(ylabel)

        # cluster order
        if allele_linkage:
            allele_order = g.dendrogram_row.reordered_ind
            allele_order = [alleles[i] for i in allele_order]
        else:
            allele_order = None
        if position_linkage:
            position_order = g.dendrogram_col.reordered_ind
            position_order = [positions[i] for i in position_order]
        else:
            position_order = None
        
        # save figure
        if figfile:
            plt.savefig(figfile, bbox_inches='tight', dpi=self.dpi)
        
        return allele_order, position_order

    
    def _motif_plot(self, alleles, motif_dict, figfile=None):
        allele_num = len(alleles)
        fig, ax = plt.subplots(allele_num, figsize=(0.8, allele_num*0.2), dpi=self.dpi)
        for i in range(allele_num):
            allele = alleles[i]
            seqlogo_df = pd.DataFrame(motif_dict[allele], columns=list(self.aa_str))
            logo = lm.Logo(seqlogo_df, ax=ax[i], color_scheme="skylign_protein")
            _ = ax[i].set_xticks([])
            _ = ax[i].set_yticks([])
            for side in ['top','bottom','left','right']:
                ax[i].spines[side].set_linewidth(0.1)

        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


    def _residue_barplot(self, arr, tag_pos, figsize=(8,3), figfile=None):
        # main figure
        fig, ax = plt.subplots(1, figsize=figsize, dpi=self.dpi)
        sns.barplot(x=list(range(self.mhc_len)), y=arr, ax=ax)
        ax.tick_params(axis='x', rotation=90)
        
        # fontsize
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        
        for item in ax.get_xticklabels():
            item.set_fontsize(self.fontsize/4)
        
        # set xtick colors
        colors = list()
        for i in range(self.mhc_len):
            if i in tag_pos:
                colors.append('red')
            else:
                colors.append('black')
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
        
        fig.tight_layout()
        
        # save figure
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')
            
    
    def _importance_stacked_barplot(self, cam_threshold, tag_pos, figsize=(8,3),
                                    xticklabels=True, yticklabels=True, figfile=None):
        # build importance dataframe, columns=['A','B','C']
        d = dict()
        for hla in ['A', 'B', 'C']:
            alleles = [i for i in self.alleles if hla in i]
            d[hla] = self._residue_importance_count(alleles, cam_threshold)
        df = pd.DataFrame(d)
        
        # figure
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        ax.margins(x=0)
        
        # stacked bar plot
        ax.bar(df.index, df['A'], color=self.color_dict['A'])
        ax.bar(df.index, df['B'], bottom=df['A'], color=self.color_dict['B'])
        ax.bar(df.index, df['C'], bottom=df['A'] + df['B'], color=self.color_dict['C'])
        
        # ticks & ticklabels
        if xticklabels:
            _ = ax.set_xticks(df.index)
            _ = ax.set_xticklabels(df.index+1, rotation=90)
            # xtick colors
            colors = list()
            for i in df.index:
                if i in tag_pos:
                    colors.append('red')
                else:
                    colors.append('black')
            for tick, color in zip(ax.get_xticklabels(), colors):
                tick.set_color(color)
        else:
            _ = ax.set_xticks([])
            _ = ax.set_xticklabels([])
        
        if yticklabels:
            _ = ax.set_ylabel('importance')
        else:
            _ = ax.set_yticks([])
            _ = ax.set_yticklabels([])
                
        # fontsize
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        
        # legend
        Abar = matplotlib.patches.Rectangle((0,0),1,1,fc=self.color_dict['A'], edgecolor='none')
        Bbar = matplotlib.patches.Rectangle((0,0),1,1,fc=self.color_dict['B'], edgecolor='none')
        Cbar = matplotlib.patches.Rectangle((0,0),1,1,fc=self.color_dict['C'], edgecolor='none')
        l = ax.legend([Abar, Bbar, Cbar], ['HLA-A', 'HLA-B', 'HLA-C'], loc=0, ncol=3, fontsize=self.fontsize)
        l.draw_frame(False)
        
        fig.tight_layout()
        
        # save figure
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')

        return df
            
            
    def _mhc_importance_polymorphism_plot(self, cam_threshold, position_dict, figsize=(3.5,3.5), s=2, figfile=None):
        # figure
        df = pd.DataFrame()
        fig, ax = plt.subplots(1, figsize=figsize, dpi=self.dpi)

        # calculate entropy
        df['polymorphism'] = -(self.mhc_seqlogo_df*np.log(self.mhc_seqlogo_df)).sum(axis=1)

        # calculate importance by HLA
        importance_counts = list()
        for hla in ['A', 'B', 'C']:
            alleles = [i for i in self.alleles if hla in i]
            importance_counts.append(self._residue_importance_count(alleles, cam_threshold))
        importance_counts = np.array(importance_counts)
        importance_count = importance_counts.max(axis=0)
        df['importance'] = importance_count

        # label
        df['label'] = 'others'
        df.loc[position_dict['res34'], 'label'] = '34-residue'
        df.loc[position_dict['selected'], 'label'] = 'selected'
        intersect = list(set(position_dict['res34']) & set(position_dict['selected']))
        df.loc[intersect, 'label'] = 'intersection'

        # plot_param
        param_dict = OrderedDict({'selected':{'color': '#ff4949', 'marker': 'o', 's': 12},
                                  'intersection': {'color': '#ff4949', 'marker': 'x', 's': 12},
                                  '34-residue': {'color': '#adb5bd', 'marker': 'x', 's': 12},
                                  'others': {'color': '#adb5bd', 'marker': 'o', 's': 12}})

        # regplot
        df = df[df['polymorphism']!=0]
        p = sns.regplot(x='importance', y='polymorphism', data=df, ax=ax, fit_reg=True, scatter_kws={'s':0})
        for label, params in param_dict.items():
            p = sns.regplot(x='importance', y='polymorphism', data=df[df['label']==label],
                            ax=ax, fit_reg=False, marker=params['marker'],
                            scatter_kws={'color':params['color'], 's':params['s'], 'linewidths': 0.1})
        '''
        # annotation
        for idx, row in df.iterrows():
            if idx in [64, 70]:
                p.text(df.loc[idx, 'importance']-0.025, df.loc[idx, 'polymorphism']-0.09, idx+1, fontsize=self.fontsize-2)
        '''
        # fontsize
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)

        # legend
        legend_list = [matplotlib.patches.Rectangle((0,0),1,1,fc='#ff4949', edgecolor='none'),
                       matplotlib.patches.Rectangle((0,0),1,1,fc='#adb5bd', edgecolor='none'),
                       plt.scatter([], [], color='black', marker='x', s=12),
                       plt.scatter([], [], color='black', marker='o', s=12)]
        label_list = ['selected', 'non-selected', '34-residue', 'non-34-residue']
        l = ax.legend(handles=legend_list, labels=label_list,
                      loc='lower left', bbox_to_anchor=(-0.2,1), ncol=2, fontsize=self.fontsize)
        l.draw_frame(True)

        # layout
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.02])
        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, ''])
        fig.tight_layout()

        # pearson correlation
        pearson, pvalue = pearsonr(df['importance'], df['polymorphism'])
        ax.text(0.05, 1.6, 'r=%.2f, p=%.2e'%(pearson, pvalue))

        # save figure
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')

        return df
    

    def _epitope_importance_polymorphism_plot(self, figsize=(3.5,3.5), figfile=None):
        # get epitope polymorphism
        peptides = self.df[self.df[self.pred_basename] > self.pred_threshold]['sequence'].to_list()
        peptides = [i[:self.submotif_len] + i[-self.submotif_len:] for i in peptides]
        seqlogo_df = lm.alignment_to_matrix(sequences=peptides, to_type="probability",
                                            characters_to_ignore=".", pseudocount=0)
        polymorphism = -(seqlogo_df*np.log(seqlogo_df)).sum(axis=1).to_numpy()

        # df for plot
        df = pd.DataFrame(index=list(range(1, 1+self.submotif_len)) + list(range(-self.submotif_len, 0)))
        df['polymorphism'] = polymorphism
        df['mask_score'] = self.epitope_mask_df.mean(axis=1)[df.index]
        df['residue_tag'] = 'other'
        df.loc[[2,-1], 'residue_tag'] = 'anchor'
        
        # plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=self.dpi)
        sns.scatterplot(data=df, x='mask_score', y='polymorphism', hue='residue_tag', ax=ax)
        for pos in [2, -1]:
            ax.text(x=df.loc[pos, 'mask_score']-0.25, y=df.loc[pos, 'polymorphism'], s='Position: {}'.format(pos))
        
        fig.tight_layout()
        
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')

        return df
            

    def _importance_venn_plot(self, position_dict, figsize=(3.5,3.5), figfile=None):
        keys = ['A','B','C','polymorphism']
        position_dict = {k: set(v) for k, v in position_dict.items() if k in keys}
        petal_labels = generate_petal_labels(position_dict.values())
        colors = [list(np.array(self._convert_color_code(self.color_dict[k]))/256) + [0.4] for k in keys]
        
        fig, ax = plt.subplots(1, figsize=figsize, dpi=self.dpi)
        draw_venn(petal_labels=petal_labels, dataset_labels=position_dict.keys(),hint_hidden=False,
                  colors=colors, figsize=figsize, fontsize=self.fontsize, legend_loc="best", ax=ax)
        
        ax.get_legend().remove()
        
        legends = [matplotlib.patches.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
        l = fig.legend(legends, keys, fontsize=self.fontsize,
                       ncol=4, loc="lower center", bbox_to_anchor=(0, 0.75, 1, 0.2),
                       columnspacing=1, handlelength=0.5, handletextpad=0.2, borderpad=0.2)
        
        fig.tight_layout()
        
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')


    """"""""""""""""""""""""""""""""""""""
    # Minor Functions
    """"""""""""""""""""""""""""""""""""""
    def _parse_mask(self, dirname, mask_type):
        masks = OrderedDict()
        for allele in os.listdir(dirname):
            if re.match(r'[ABC][0-9]+', allele):
                if not os.path.isfile('%s/%s/record.npy'%(dirname, allele)):
                    continue
                if mask_type == 'mhc':
                    masks[self._convert_allele(allele)] \
                    = np.load('%s/%s/record.npy'%(dirname, allele), allow_pickle=True)[()]['mhc_masks'].mean(axis=0)
                else:
                    masks[self._convert_allele(allele)] \
                    = np.load('%s/%s/record.npy'%(dirname, allele), allow_pickle=True)[()]['epitope_masks'].mean(axis=0)
        return masks


    def _parse_motif(self, basename, threshold, sample_num):
        motifs = OrderedDict()
        for i in range(self.allele_num):
            allele = self.alleles[i]
            seqs = self.df.loc[(self.df['mhc']==allele) & (self.df[basename] >= threshold), 'sequence']
            if len(seqs) >= sample_num:
                seqs = seqs.apply(lambda x: x[:self.submotif_len] + x[-self.submotif_len:])
                temp_df = pd.DataFrame(columns=list(self.aa_str))
                seqlogo_df = lm.alignment_to_matrix(sequences=seqs, to_type="information", characters_to_ignore="XU")
                temp_df = pd.concat([temp_df, seqlogo_df], axis=0)
                temp_df = temp_df.fillna(0.0)
                motifs[allele] = temp_df.to_numpy()
        return motifs


    def _residue_importance_count(self, alleles, cam_threshold):
        importance_count = np.array([0]*self.mhc_len)
        for allele in alleles:
            importance_count[self.allele_mask_df[allele] > cam_threshold] += 1
        return importance_count / len(alleles)


    def _mhc_seqlogo_df(self, alleles, positions):
        seqs = list()
        for allele in alleles:
            seqs.append(''.join(self.mhc_seq_dict[allele][j] for j in positions))
        temp_df = pd.DataFrame(columns=list(self.aa_str))
        seqlogo_df = lm.alignment_to_matrix(sequences=seqs, to_type="probability",
                                            characters_to_ignore=".", pseudocount=0)
        temp_df = pd.concat([temp_df, seqlogo_df], axis=0)
        temp_df = temp_df.fillna(0.0)
        return temp_df


    def _select_residue(self, cam_threshold, importance_threshold):
        importance_positions = dict()
        importance_position_set = set()
        importance_positions['res34'] = self.res34_pos

        # by HLA
        for hla in ['A', 'B', 'C']:
            alleles = [i for i in self.alleles if hla in i]
            importance_count = self._residue_importance_count(alleles, cam_threshold)
            pos = list(map(int, np.where(importance_count > importance_threshold)[0]))
            importance_positions[hla] = pos
            importance_position_set = importance_position_set | set(pos)
        
        # polymorphism
        polymorphism_position = list(map(int,self.mhc_seqlogo_df[~(self.mhc_seqlogo_df.max(axis=1)==1)].index))
        importance_positions['polymorphism'] = sorted(polymorphism_position)
        importance_position_set = importance_position_set & set(polymorphism_position)
        
        # final
        importance_position = sorted(list(importance_position_set))
        importance_positions['selected'] = importance_position
        
        return importance_positions


    def _convert_allele(self, allele):
        if re.match(r'[ABC][0-9]+', allele):
            return allele[0] + '*' + allele[1:-2] + ':' + allele[-2:]
        elif re.match(r'[ABC]\*[0-9]+\:[0-9]+', allele):
            return allele
        
    
    def _convert_color_code(self, code):
        return tuple(int(code[i:i+2], 16) for i in (1, 3, 5))
