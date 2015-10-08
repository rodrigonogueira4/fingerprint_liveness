import numpy as np
from skimage.feature import local_binary_pattern   
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

class LBP():
    
    n_tiles = [1, 1]
    radius = 1
    method = 'default'
    hist = True
    
    pmask_ror = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                     1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.bool) # tirado o ultimo codigo (fundo)
    pmask_uniform = np.array([1,1,1,1,1,1,1,1,0,1]).astype(np.bool) # tirado o antepenultimo (fundo) 
    #print 'pmask do uniform esta errado!!'
    #pmask_uniform = np.array([0,0,1,1,1,1,1,1,0,0]).astype(np.bool) # tirando mais dimensoes
    
        
    def extract(self, img):
        n_points = 8 * self.radius
        if self.hist:
            tile_size = (np.rint(np.asarray(img.shape)/self.n_tiles)).astype(np.int32) 
            #"""
            if self.method == 'default': 
                nsize = 255
                pmask = np.ones((256,)).astype(np.bool)
                pmask[255] = 0
            elif self.method == 'ror': 
                nsize = self.pmask_ror.sum()
                pmask = self.pmask_ror
            elif self.method == 'uniform': 
                nsize = self.pmask_uniform.sum()
                pmask = self.pmask_uniform
            else:
                nsize = 256
            
            """
            if self.method == 'default': 
                nsize = 256
                pmask = np.ones((256,)).astype(np.bool)
            elif self.method == 'ror': 
                nsize = self.pmask_ror.sum()
                pmask = self.pmask_ror
            elif self.method == 'uniform': 
                nsize = n_points +2
                pmask = np.ones((nsize,)).astype(np.bool)
            else: nsize = 256
            """
            
            histconcat = np.empty((self.n_tiles[0],self.n_tiles[1], nsize))
       
            for j in np.arange(np.int(self.n_tiles[0])):
                for i in np.arange(np.int(self.n_tiles[1])):
                    tile = img[j*tile_size[0]:(j+1)*tile_size[0],i*tile_size[1]:(i+1)*tile_size[1]]
                    tilelbp = local_binary_pattern(tile,P=n_points, R=self.radius, method=self.method).astype(np.uint8)
                    tilehistlbp = np.bincount(np.ravel(tilelbp))#equivalent to histogram
                    tilehistlbp = np.concatenate((tilehistlbp, np.zeros((pmask.shape[0]-tilehistlbp.shape[0]),dtype = np.int))) #add the remaining bins in case they exists
                    th = tilehistlbp[pmask]
                    histconcat[j,i] = th
            return histconcat.reshape(-1)
        else:
            tilelbp = local_binary_pattern(img,P=n_points, R=self.radius, method=self.method).astype(np.uint8)
    

    def hog(self, image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(3, 3), visualise=False, normalise=False):
        """Extract Histogram of Oriented Gradients (HOG) for a given image.
    
        Compute a Histogram of Oriented Gradients (HOG) by
    
            1. (optional) global image normalisation
            2. computing the gradient image in x and y
            3. computing gradient histograms
            4. normalising across blocks
            5. flattening into a feature vector
    
        Parameters
        ----------
        image : (M, N) ndarray
            Input image (greyscale).
        orientations : int
            Number of orientation bins.
        pixels_per_cell : 2 tuple (int, int)
            Size (in pixels) of a cell.
        cells_per_block  : 2 tuple (int,int)
            Number of cells in each block.
        visualise : bool, optional
            Also return an image of the HOG.
        normalise : bool, optional
            Apply power law compression to normalise the image before
            processing.
    
        Returns
        -------
        newarr : ndarray
            HOG for the image as a 1D (flattened) array.
        hog_image : ndarray (if visualise=True)
            A visualisation of the HOG image.
    
        References
        ----------
        * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    
        * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
          Human Detection, IEEE Computer Society Conference on Computer
          Vision and Pattern Recognition 2005 San Diego, CA, USA
    
        """
        image = np.atleast_2d(image)
    
        """
        The first stage applies an optional global image normalisation
        equalisation that is designed to reduce the influence of illumination
        effects. In practice we use gamma (power law) compression, either
        computing the square root or the log of each colour channel.
        Image texture strength is typically proportional to the local surface
        illumination so this compression helps to reduce the effects of local
        shadowing and illumination variations.
        """
    
        if image.ndim > 2:
            raise ValueError("Currently only supports grey-level images")
    
        if normalise:
            image = sqrt(image)
    
        """
        The second stage computes first order image gradients. These capture
        contour, silhouette and some texture information, while providing
        further resistance to illumination variations. The locally dominant
        colour channel is used, which provides colour invariance to a large
        extent. Variant methods may also include second order image derivatives,
        which act as primitive bar detectors - a useful feature for capturing,
        e.g. bar like structures in bicycles and limbs in humans.
        """
    
        if image.dtype.kind == 'u':
            # convert uint image to float
            # to avoid problems with subtracting unsigned numbers in np.diff()
            image = image.astype('float')
    
        gx = np.zeros(image.shape)
        gy = np.zeros(image.shape)
        gx[:, :-1] = np.diff(image, n=1, axis=1)
        gy[:-1, :] = np.diff(image, n=1, axis=0)
    
        """
        The third stage aims to produce an encoding that is sensitive to
        local image content while remaining resistant to small changes in
        pose or appearance. The adopted method pools gradient orientation
        information locally in the same way as the SIFT [Lowe 2004]
        feature. The image window is divided into small spatial regions,
        called "cells". For each cell we accumulate a local 1-D histogram
        of gradient or edge orientations over all the pixels in the
        cell. This combined cell-level 1-D histogram forms the basic
        "orientation histogram" representation. Each orientation histogram
        divides the gradient angle range into a fixed number of
        predetermined bins. The gradient magnitudes of the pixels in the
        cell are used to vote into the orientation histogram.
        """
    
        magnitude = sqrt(gx**2 + gy**2)
        orientation = arctan2(gy, gx) * (180 / pi) % 180
    
        sy, sx = image.shape
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
    
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    
        # compute orientations integral images
        orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
        subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]
        for i in range(orientations):
            #create new integral image for this orientation
            # isolate orientations in this range
    
            temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                                orientation, -1)
            temp_ori = np.where(orientation >= 180 / orientations * i,
                                temp_ori, -1)
            # select magnitudes for those orientations
            cond2 = temp_ori > -1
            temp_mag = np.where(cond2, magnitude, 0)
    
            temp_filt = uniform_filter(temp_mag, size=(cy, cx))
            orientation_histogram[:, :, i] = temp_filt[subsample]
    
        # now for each cell, compute the histogram
        hog_image = None
        
        if visualise:
            from skimage import draw
    
            radius = min(cx, cy) // 2 - 1
            hog_image = np.zeros((sy, sx), dtype=float)
            for x in range(n_cellsx):
                for y in range(n_cellsy):
                    for o in range(orientations):
                        centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                        dx = radius * cos(float(o) / orientations * np.pi)
                        dy = radius * sin(float(o) / orientations * np.pi)
                        rr, cc = draw.bresenham(int(centre[0] - dx),
                                                int(centre[1] - dy),
                                                int(centre[0] + dx),
                                                int(centre[1] + dy))
                        hog_image[rr, cc] += orientation_histogram[y, x, o]
    
        """
        The fourth stage computes normalisation, which takes local groups of
        cells and contrast normalises their overall responses before passing
        to next stage. Normalisation introduces better invariance to illumination,
        shadowing, and edge contrast. It is performed by accumulating a measure
        of local histogram "energy" over local groups of cells that we call
        "blocks". The result is used to normalise each cell in the block.
        Typically each individual cell is shared between several blocks, but
        its normalisations are block dependent and thus different. The cell
        thus appears several times in the final output vector with different
        normalisations. This may seem redundant but it improves the performance.
        We refer to the normalised block descriptors as Histogram of Oriented
        Gradient (HOG) descriptors.
        """
    
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                      by, bx, orientations))
    
        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[y:y + by, x:x + bx, :]
                eps = 1e-5
                normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)
    
        """
        The final step collects the HOG descriptors from all blocks of a dense
        overlapping grid of blocks covering the detection window into a combined
        feature vector for use in the window classifier.
        """
    
        if visualise:
            return normalised_blocks.ravel(), hog_image
        else:
            return normalised_blocks.ravel()