from sklearn.cluster import KMeans
from sklearn.preprocessing import scale 
from copy import deepcopy
from cluster import dti_cluster
import nibabel as nib 
import numpy as np
import pandas as pd 
import h5sparse


def mask_img(img,val):
    """masks an image at a given value - borrowed from de la Vega"""
    img = deepcopy(img)
    data = img.get_data()
    data[:] = np.round(data)
    data[data!=val] = 0
    data[data==val] = 1
    return img


def threshold_image(img,thr):
    if isinstance(img, str):
        img = nib.load(img)
    data = img.get_data()
    data[np.where(data<thr)] = 0
    return img
    


class Query(object):
    
    def __init__(self, dataset, mat=None):
        
        if isinstance(dataset,str):
            self.dataset = h5sparse.File(dataset)
        else:
            self.dataset = dataset
        if mat is not None:
            self.data = mat#self.dataset['sparse/matrix'][:]
        else:
            self.data = self.dataset['sparse/matrix'][:]
        self.reference = self.dataset['reference/data'][:]
        self.x = self.dataset['reference/x'][:] # convert to dask?
        self.y = self.dataset['reference/y'][:]
        self.z = self.dataset['reference/z'][:]
        self.voxel_num = self.dataset['reference/voxel_num'][:]
        
    
    def reduce_matrix(self, threshold):
        l = nib.load('../masks/l_white_matter.nii.gz').get_data()
        l[np.where(l<threshold)] = 0
        r = nib.load('../masks/r_white_matter.nii.gz').get_data()
        r[np.where(r<threshold)] = 0
        white = np.zeros([91,109,91])
        white[np.where(l>0)] = 1
        white[np.where(r>0)] = 1
        
        to_remove = np.array(self.reference[np.where(white>0)])
        to_keep = np.delete(np.arange(0,self.data.shape[0]),to_remove)
        
        self.data = self.data[:,to_keep]
        return
        
    
    def init_nifti(self,img_data,header=None,affine=None):
        """for saving images """
        header.set_data_dtype(img_data.dtype) 
        header['cal_max'] = img_data.max()
        header['cal_min'] = img_data.min()
        return nib.nifti1.Nifti1Image(img_data, affine=affine,header=header)
    
    
    def get_roi_matrix(self,img):
        """a function for generating a connectivity matrix between every voxel in 
        an roi and every voxel in the gray-matter mask"""
        
        if isinstance(img, str):
            img = nib.load(img)
        img_data = img.get_data()
        roi_coords = self.reference[img_data > 0].astype(int) # finding the roi
        roi_coords.sort() # this is probably not needed
        #sliced = self.dataset['sparse/matrix'][roi_coords.min():roi_coords.max()+1]
        #return sliced[roi_coords - roi_coords.min(),:]
        return self.data[roi_coords,:]
    
    def cluster_roi(self,img,n_clusters=3):
        """wrapper for dti_cluster """
        if isinstance(img,str):
            img = nib.load(img)
        roi_coords = self.reference[img.get_data()>0].astype(int)
        roi_coords.sort()
        labels = dti_cluster(self.get_roi_matrix(img),n_clusters=n_clusters)
        clusters = np.zeros([91,109,91])
        coords_x,coords_y,coords_z = self.x[roi_coords],self.y[roi_coords],self.z[roi_coords]
        clusters[coords_x,coords_y,coords_z] = labels
        return self.init_nifti(clusters,img.header,img.affine)

    
    
    def sum_streamline_count(self,img, output='nifti'):
        """generates an array of the streamline count between a cluster and each voxel in the rest of the brain """
        
        if isinstance(img,str):
            img = nib.load(img)
        mat = self.get_roi_matrix(img)
        vals = mat.toarray().sum(axis=0)   # get the total streamline count by summing the values of each column
        vals = np.delete(vals,0) # the zero voxel doesn't exist so we'll remove it
        
        if output=='vector':
            return vals
        if output=='nifti':
            file = np.zeros([91,109,91])
            file[self.x,self.y,self.z] = vals
            return self.init_nifti(file,affine=img.affine,header=img.header)
        
    
    # not finished
    def connectivity_contrast(self,img):
    
            
        def get_comp_matrix(img,i):
            coords = self.reference[img.get_data()>0]
            coords = np.delete(coords, self.reference[img.get_data()==i])
            return self.data[coords,:]
        
        if isinstance(img,str):
            img = nib.load(img)
            
        images = []
        for i in range(1,img.get_data().round().astype(int).max() + 1): 
            cluster = mask_img(img,i)
            roi_mat = self.get_roi_matrix(cluster)
            comp_mat = get_comp_matrix(img,i)
            cluster_total = roi_mat.toarray().sum(axis=0)/roi_mat.shape[0]
            comp_total = comp_mat.toarray().sum(axis=0)/comp_mat.shape[0]
            contrast = cluster_total - comp_total
            contrast[np.where(contrast<0)] = 0
            contrast = np.delete(contrast,0)
            stat_map = np.zeros([91,109,91])
            stat_map[self.x,self.y,self.z] = contrast
            images.append(self.init_nifti(stat_map,header=cluster.header,affine=cluster.affine))
        return images
    
    
        
    
    
    def roi_similarity(self,img):
        """for comparing the similarity of connectivity distributions bewteen different clusters.
        returns a correlation matrix """
        if isinstance(img,str):
            img = nib.load(img)
        connectivity_vectors = []
        for i in range(1,img.get_data().max().astype(int) + 1):
            cluster = mask_img(img,i)
            connectivity_vectors.append(self.sum_streamline_count(cluster, output='vector'))
        mat = np.vstack((connectivity_vectors[:]))
        
        CC = np.corrcoef(mat)
        
        return np.nan_to_num(CC)

                
    
    def connections_to_targets(self,img,targets, normalize=False, labels=None, as_df=False):
        """given an roi it returns a dataframe of connections between the roi and each of the targets """
        if isinstance(img,str):
            img = nib.load(img)
        if isinstance(targets,str):
            targets = nib.load(targets)
        stat_map = self.sum_streamline_count(img,output='nifti')
        stat_data = stat_map.get_data()
        target_data = targets.get_data().round()    
        connections = np.array([stat_data[target_data==i].sum() for i in range(1,target_data.max().astype(int) + 1)])
        
        if as_df:
            df = pd.DataFrame()
            #if normalize:
            totals = np.array([target_data[target_data==i].shape[0] for i in range(1,target_data.max().astype(int) + 1)])
            df['connections'] = pd.Series(connections)
            df['normalized_connections'] = pd.Series(np.array(connections)/np.array(totals))
            df['percent'] = pd.Series(df['normalized_connections']/df['normalized_connections'].sum())
        
            if labels is not None:
                df['labels'] = pd.Series(labels)
            
            
        
            return df
        else:
            return connections

    # work in progress
    def network(self,regions,normalize=False,labels=None):
        """returns a numpy array of streamline counts between all regions"""
        # better performance with smaller N. Scales O(N**2)
        if isinstance(regions, str):
            regions = nib.load(regions)
        total = []
        for i in range(1,regions.get_data().max().astype(int) + 1):
            roi = mask_img(regions,i)
            total.append(self.connections_to_targets(roi,regions))
        
        #normalize = np.array([network[i,:]/network[i,:].sum() for i in range(48)])
        #total = np.array(total)
        #if normalize:
        return np.array(total)
        
        
        # if no labels are provided, regions will have numeric labels
        #if labels is None:
        #    labels = np.arange(region_data.max())
        
       # df = pd.DataFrame(columns=labels,index=labels)
        
        #for i in range(1,region_data.max()+1):
         #   region = mask_img(regions)
         #   region_connections = self.connections_to_targets(region,regions)
            
            
