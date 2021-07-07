import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm
import base64
import io

def get_index_image():
    with open('static/images/overview.png', 'rb') as f:
        overview = base64.b64encode(f.read()).decode('utf-8')
    with open('static/images/application.png', 'rb') as f:
        application = base64.b64encode(f.read()).decode('utf-8')

    return overview, application

class Interpretation():
    def __init__(self, interpretation_file):
        self.aa_str = 'ACDEFGHIKLMNPQRSTVWY'
        self.sub_motif_len = 4
        self.dpi = 200
        self.fontsize = 8
        
        self.interp_dict = pickle.load(open(interpretation_file, 'rb'))
        self.positions = self.interp_dict['important_positions']
        self.mhc_dict = self.interp_dict['seq']
        self.motif_dict = self.interp_dict['motif']

    def get_overview_image(self, allele):
        hla = allele.split('*')[0]
        with open('static/images/interpretation_{}.png'.format(hla), 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return image_data

    def get_detail_image(self, allele):

        result = {
            'N': {'hyper_motif': '', 'allele_signature': '', 'sub_motif': '', 'highlighted_allele_signature': '' },
            'C': {'hyper_motif': '', 'allele_signature': '', 'sub_motif': '', 'highlighted_allele_signature': '' }
        }
        hla = allele.split('*')[0]
        motif_df = pd.DataFrame(self.motif_dict[allele], columns=list(self.aa_str))
        allele_df = self._get_allele_seqlogo(allele)
        
        for side in ['N', 'C']:
            # cluster
            cluster = self.interp_dict['cluster']['%s_%s'%(hla, side)][allele]
            
            # sub motif
            if side == 'N':
                sub_motif_df = motif_df.iloc[:4]
            else:
                sub_motif_df = motif_df.iloc[-4:].reset_index(drop=True)
                
            # sub-motif plot
            fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=self.dpi)
            self._motif_plot(sub_motif_df, side, ax)
            fig.tight_layout()
            pic_IObytes = io.BytesIO()
            fig.savefig(pic_IObytes, format='png')
            pic_IObytes.seek(0)
            result[side]['sub_motif'] = base64.b64encode(pic_IObytes.read()).decode('utf-8')
            
            # check cluster
            if cluster not in self.interp_dict['hyper_motif']['%s_%s'%(hla, side)].keys():
                continue
                
            # hyper-motif plot
            with open('static/images/hyper_motif/%s_%s_%d.png'%(hla, side, cluster), 'rb') as f:
                result[side]['hyper_motif'] = base64.b64encode(f.read()).decode('utf-8')
            '''
            hyper_motif = self.interp_dict['hyper_motif']['%s_%s'%(hla, side)][cluster]
            hyper_motif = pd.DataFrame(hyper_motif, columns=list(self.aa_str))
            fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=self.dpi)
            self._motif_plot(hyper_motif, side, ax)
            fig.tight_layout()
            pic_IObytes = io.BytesIO()
            fig.savefig(pic_IObytes, format='png')
            pic_IObytes.seek(0)
            result[side]['hyper_motif'] = base64.b64encode(pic_IObytes.read()).decode('utf-8')
            '''
            
            # allele signature plot
            with open('static/images/allele_signature/%s_%s_%d.png'%(hla, side, cluster), 'rb') as f:
                result[side]['allele_signature'] = base64.b64encode(f.read()).decode('utf-8')
            
            allele_signature = self.interp_dict['allele_signature']['%s_%s'%(hla, side)][cluster]
            allele_signature = pd.DataFrame(allele_signature, columns=list(self.aa_str))
            '''
            fig, ax = plt.subplots(1, 1, figsize=(10, 2), dpi=self.dpi)
            self._mhcseq_plot(allele_signature, ax)
            fig.tight_layout()
            pic_IObytes = io.BytesIO()
            fig.savefig(pic_IObytes, format='png')
            pic_IObytes.seek(0)
            result[side]['allele_signature'] = base64.b64encode(pic_IObytes.read()).decode('utf-8')
            '''
            
            # highlighted allele signature plot
            allele_df[allele_df > 0] = 1
            allele_signature[allele_signature < 0] = 0
            fig, ax = plt.subplots(1, 1, figsize=(10, 2), dpi=self.dpi)
            self._mhcseq_plot(allele_df * allele_signature, ax)
            fig.tight_layout()
            pic_IObytes = io.BytesIO()
            fig.savefig(pic_IObytes, format='png')
            pic_IObytes.seek(0)
            result[side]['highlighted_allele_signature'] = base64.b64encode(pic_IObytes.read()).decode('utf-8')

        return result

    def _get_allele_seqlogo(self, allele):
        seq = self.mhc_dict[allele]
        seq = ''.join([seq[i] for i in self.positions])
        seqlogo_df = lm.alignment_to_matrix(sequences=[seq], to_type='counts')
        df = pd.DataFrame(columns=list(self.aa_str))
        return pd.concat([df, seqlogo_df], axis=0)[list(self.aa_str)].fillna(0)


    def _motif_plot(self, seqlogo_df, side, ax, ylim=4, title=None, turn_off_label=False):
        if side == 'N':
            xticklabels = list(range(1, self.sub_motif_len+1))
        else:
            xticklabels = list(range(-self.sub_motif_len, 0))
        logo = lm.Logo(seqlogo_df, color_scheme='skylign_protein', ax=ax)
        _ = ax.set_xticks(list(range(len(xticklabels))))
        _ = ax.set_xticklabels(xticklabels)
        _ = ax.set_ylim(0,ylim)
        _ = ax.set_title(title)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)

        _ = ax.set_yticks([])
        _ = ax.set_yticklabels([])
            
        if turn_off_label:
            _ = ax.set_xticks([])
            _ = ax.set_xticklabels([])
            _ = ax.set_title(None)


    def _mhcseq_plot(self, seqlogo_df, ax, ylim=1, title=None, turn_off_label=False):
        logo = lm.Logo(seqlogo_df, color_scheme='skylign_protein', ax=ax)
        _ = ax.set_ylim(0, ylim)
        _ = ax.set_xticks(range(len(self.positions)))
        _ = ax.set_xticklabels([i+1 for i in self.positions], rotation=90)
        _ = ax.set_title(title, loc='left')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)

        _ = ax.set_yticks([])
        _ = ax.set_yticklabels([])
            
        if turn_off_label:
            _ = ax.set_xticks([])
            _ = ax.set_xticklabels([])
            _ = ax.set_title(None)