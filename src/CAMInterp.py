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
from scipy.cluster import hierarchy
from util import *
import warnings
warnings.filterwarnings('ignore')


class CAMInterp():
    def __init__(self, mhc_seq_filename, mask_dirname, df_filename, output_dir,
                 pred_basename='bind', pred_threshold=1, mhc_len=182, min_sample_num=100, submotif_len=4):
        self.aa_str = 'ACDEFGHIKLMNPQRSTVWY'
        
        self.mhc_len = mhc_len
        self.res34_pos = [6, 8, 23, 44, 58, 61, 62, 65, 66, 68, 69, 72, 73, 75, 76, 79, 80, 83, 94,
                          96, 98, 113, 115, 117, 142, 146, 149, 151, 155, 157, 158, 162, 166, 170]
        
        self.color_dict = {'A': '#DACC47', 'B': '#B1DEC9', 'C': '#FFBB99', 'polymorphism': '#875A85'}
        
        self.dpi = 600
        self.fontsize = 10

        self.min_sample_num = min_sample_num
        self.submotif_len = submotif_len

        self.output_dir = output_dir

        # mhc_seq_dict
        self.mhc_seq_dict = json.load(open(mhc_seq_filename, 'r'))

        # mask_df
        if type(mask_dirname) == list:
            alleles = [self._convert_allele(i) for i in os.listdir(mask_dirname[0])]
            self.mask_df = pd.DataFrame(columns=alleles, index=range(self.mhc_len), data=0)
            self.mask_df.loc['count'] = 0
            for i in range(len(mask_dirname)):
                temp_df = pd.DataFrame(self._parse_mask(mask_dirname[i]))
                self.mask_df.loc[temp_df.index, temp_df.columns] += temp_df
                self.mask_df.loc['count', temp_df.columns] += 1
            self.mask_df = self.mask_df.loc[:, self.mask_df.loc['count'] != 0]
            self.mask_df.loc[range(mhc_len)] /= self.mask_df.loc['count']
            self.mask_df = self.mask_df.drop('count')
        else:
            self.mask_df = pd.DataFrame(self._parse_mask(mask_dirname))
        
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


    def ResidueAnalysis(self, cam_threshold, importance_threshold):
        # mean plot
        self._residue_barplot(self.mask_df.mean(axis=1), self.res34_pos,
                              figfile='%s/CAMmean.png'%self.output_dir)

        # importance plot
        importance_count = self._residue_importance_count(self.alleles, cam_threshold)
        self._residue_barplot(importance_count, self.res34_pos,
                              figfile='%s/CAMimportance.png'%self.output_dir)
    
        # important residues - stacked plot
        self._importance_stacked_barplot(cam_threshold, self.res34_pos,
                                         xticklabels=False, yticklabels=False,
                                         figfile='%s/CAMimportanceStacked.png'%self.output_dir)
        
        # important residues
        residue_dict = self._select_residue(cam_threshold, importance_threshold)
        json.dump(residue_dict, open('%s/ResidueSelection.json'%self.output_dir, 'w'))
        
        # venn diagram of residue selection
        self._importance_venn_plot(residue_dict,
                                   figfile='%s/ResidueSelectionVenn.png'%self.output_dir)

        # correlation between residue importance and mhc sequence entropy
        # entropy = sigma(probability**2)
        self._importance_polymorphism_plot(cam_threshold, self.res34_pos,
                                           figfile='%s/ImportanceEntropyCorrelation.png'%self.output_dir)


    def ClusterAnalysis(self, method, metric, plot_each_mhc=True):
        # all alleles
        alleles = self.alleles
        allele_order, _ = self._mask_clustering_plot(alleles, method=method, metric=metric,
                                                     xticklabels=False, yticklabels=False,
                                                     row_colors=True, figsize=(8,4),
                                                     figfile='%s/AlleleCAMcluster_all.png'%self.output_dir)

        # each MHCI
        if not plot_each_mhc:
            return
        for hla in ['A', 'B', 'C']:
            alleles = [i for i in self.motif_dict.keys() if hla in i]
            allele_order, _ = self._mask_clustering_plot(alleles, method=method, metric=metric,
                                                         figfile='%s/AlleleCAMcluster_%s.png'%(self.output_dir, hla))
            self._motif_plot(allele_order, self.motif_dict, figfile='%s/AlleleMotif_%s.png'%(self.output_dir, hla))


    """"""""""""""""""""""""""""""""""""""
    # Plots
    """"""""""""""""""""""""""""""""""""""
    def _mask_clustering_plot(self, alleles, positions=None,
                              method='average', metric='euclidean',
                              allele_linkage=True, position_linkage=False,
                              row_colors=False, xticklabels=True, yticklabels=True,
                              figsize=(8, 4), figfile=None):
        # residue positions
        if not positions:
            positions = list(range(self.mhc_len))
        df = self.mask_df.iloc[positions][alleles].T
        
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
                           cbar_pos=None,
                           dendrogram_ratio=0.1,
                           colors_ratio=0.02,
                           xticklabels=xticklabels,
                           yticklabels=yticklabels,
                           figsize=figsize)
        
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
        
        if not yticklabels:
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
            
            
    def _importance_polymorphism_plot(self, cam_threshold, tag_pos, figsize=(3.5,3.5), s=2, figfile=None):
        # figure
        df = pd.DataFrame()
        fig, ax = plt.subplots(1, figsize=figsize, dpi=self.dpi)
        ax.margins(x=1)
        
        # calculate importance and entropy
        df['polymorphism'] = -(self.mhc_seqlogo_df*np.log(self.mhc_seqlogo_df)).sum(axis=1)
        importance_count = self._residue_importance_count(self.alleles, cam_threshold)
        df['importance'] = importance_count
        df['color'] = '#91bfdb'
        df.loc[tag_pos, 'color'] = '#fc8d59'
        sns.regplot(x='importance', y='polymorphism', data=df, ax=ax,
                    scatter_kws={'color':list(df['color']), 's':s})
        
        # fontsize
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        
        # legend
        legend_a = matplotlib.patches.Rectangle((0,0),1,1,fc='#fc8d59', edgecolor='none')
        legend_b = matplotlib.patches.Rectangle((0,0),1,1,fc='#91bfdb', edgecolor='none')
        l = ax.legend([legend_a, legend_b], ['34-residue', 'others'],
                      loc='lower left', bbox_to_anchor=(0,1), ncol=2, fontsize=self.fontsize)
        l.draw_frame(True)
        
        fig.tight_layout()
        
        # save figure
        if figfile:
            fig.savefig(figfile, bbox_inches='tight')
            
            
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
    def _parse_mask(self, dirname):
        masks = OrderedDict()
        for allele in os.listdir(dirname):
            if re.match(r'[ABC][0-9]+', allele):
                if not os.path.isfile('%s/%s/record.npy'%(dirname, allele)):
                    continue
                masks[self._convert_allele(allele)] \
                = np.load('%s/%s/record.npy'%(dirname, allele), allow_pickle=True)[()]['mhc_masks'].mean(axis=0)
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
            importance_count[self.mask_df[allele] > cam_threshold] += 1
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