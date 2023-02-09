import os

def get_colors(conv_values, cmap_name='nipy_spectral', ctype='RGB', range=False):
    '''
    Returns a dict of RGB colors based on CONV_LU_V* values and matplotlib colormaps:
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    '''
    cmap = cm.get_cmap('nipy_spectral')
    norm = mcolors.Normalize(vmin=min(conv_values), vmax=max(conv_values)) if not range else mcolors.Normalize(0,
                                                                                                               len(conv_values))
    colors = []
    if ctype == 'RGB':
        for i, val in enumerate(conv_values):
            if not range:
                colors.append(list(np.array(cmap(norm(val))[:3]) * 255))
            else:
                colors.append(list(np.array(cmap(norm(i))[:3]) * 255))
    elif ctype == 'RGBA':
        for i, val in enumerate(conv_values):
            if not range:
                colors.append(cmap(norm(val)))
            else:
                colors.append(cmap(norm(i)))
    return dict(zip(conv_values, colors))


def colors2(conv_values, cmap_name='nipy_spectral', ctype='RGB', range=False):
    '''
    Returns a dict of RGB colors based on CONV_LU_V* values and matplotlib colormaps:
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    '''
    cmap = cm.get_cmap('nipy_spectral')
    norm = mcolors.Normalize(vmin=min(conv_values), vmax=max(conv_values)) if not range else mcolors.Normalize(0,
                                                                                                               len(conv_values))
    colors = []
    if ctype == 'RGB':
        for i, val in enumerate(conv_values):
            if not range:
                colors.append(list(np.array(cmap(norm(val))[:3]) * 255))
            else:
                colors.append(list(np.array(cmap(norm(i))[:3]) * 255))
    elif ctype == 'RGBA':
        for i, val in enumerate(conv_values):
            if not range:
                colors.append(cmap(norm(val)))
            else:
                colors.append(cmap(norm(i)))
    return dict(zip(conv_values, colors))


def get_img_path(url_path):
    '''
    returns the image and path based on drive link or path
    '''
    path = get_path(url_path)
    return cv2.imread(path), path


def predictions(path, resize=True):
    '''
    returns out of model.predict_segmantations()
    : param resize: boolean, True to resize out to img scale
    '''
    out = model.predict_segmentation(path)
    if resize:
        img = cv2.imread(path)
        shape = img.shape
        return cv2.resize(out, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        return out


def inverse_CONV_LU(CONV_LU):
    '''
    returns a dict with values as keys. POL labels are replaced by the
    '''
    df = pd.DataFrame.from_dict(CONV_LU, orient='index').reset_index()
    labels = df['index'].tolist()
    df['POL'] = ['_'.join(val.split('_')[:2]) in labels if 'POL' in val else False for val in df['index']]
    df.drop(df[df.POL].index, inplace=True)
    df_g = df.groupby([0]).count()
    drop_list = []
    for i in df_g[df_g['index'] > 1].index:
        j = int(input(f'Introduce the proper engin tag by index\n{df[df[0] == i]}\n'))
        drop_list += df[(df[0] == i) & (df.index != j)].index.tolist()
    df.drop(drop_list, inplace=True)
    return df[[0, 'index']].set_index(0).to_dict()['index']


def d_min(x_y, w_h, i):
    d1 = x_y - x_y[i]
    d2 = d1 + w_h
    d3 = d1 - w_h[i]
    d4 = d2 - w_h[i]
    df = pd.DataFrame([d1.abs(), d2.abs(), d3.abs(), d4.abs()]).T
    df.columns = ['d1', 'd2', 'd3', 'd4']
    return df.min(axis=1)


def check_no_intersection(df, i):
    check_x = ((df.x < df.x[i]) & (df.x + df.w < df.x[i]) | (df.x > df.x[i]) & (df.x > df.x[i] + df.w[i]))
    check_y = ((df.y < df.y[i]) & (df.y + df.h < df.y[i]) | (df.y > df.y[i]) & (df.y > df.y[i] + df.h[i]))
    return check_x, check_y


def bbox_min_dist(bboxs):
    '''
    returns a dataframe of minimum distance between bboxes
    it is assumed that x and y has the same scale 1:1 (see normilize_out)
    '''
    df = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
    dist = []
    for i in df.index:
        dx_min = d_min(df.x, df.w, i)
        dy_min = d_min(df.y, df.h, i)
        check_x, check_y = check_no_intersection(df, i)
        check = check_x * check_y
        df.at[check, 'dist'] = (dx_min[check] ** 2 + dy_min[check] ** 2) ** 0.5
        df.at[~check, 'dist'] = 0
        df.at[~check & check_x, 'dist'] = dx_min[~check & check_x]
        df.at[~check & check_y, 'dist'] = dy_min[~check & check_y]
        dist.append(df.dist.to_list())
    return pd.DataFrame(dist)


def normalize_out(out, rate_x=1, rate_y=1):
    '''
    retrunrs a normalized out, where horizontal and vertical pixels are in
    the same unit-length.
    '''
    height, width = out.shape
    n_height = height * rate_y
    n_width = width * rate_x
    out = cv2.resize(out, dsize=(int(n_width), int(n_height)), interpolation=cv2.INTER_NEAREST)
    return out


def get_big_bbox_from_conts(conts):
    data = []
    for cont in conts:
        data.append(cv2.boundingRect(cont))
    df = pd.DataFrame(data, columns=['x', 'y', 'w', 'h'])
    return get_big_bbox(df)  # *


def get_big_bbox(df_bboxs):
    x_min = df_bboxs.x.min()
    y_min = df_bboxs.y.min()
    x_max = (df_bboxs.x + df_bboxs.w).max()
    y_max = (df_bboxs.y + df_bboxs.h).max()
    w = x_max - x_min
    h = y_max - y_min
    return [x_min, y_min, w, h]


def merge_conts_by_bbox_min_dist(conts, dist_threshold=10):
    '''
    conts:          contornos from cv2.contours
    dist_threshold: distance of acceptance two or more bboxes can be marged

    returns:        dataframe with marged contours and bboxs, big bbox per set,
                    and properties such as: area_px, perimeter_px,
    '''
    bboxs = []
    properties = []
    for con in conts:
        bboxs.append(cv2.boundingRect(con))
        properties.append([cv2.contourArea(con), cv2.arcLength(con, True)])  # contour area and perimeter
    df_bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
    df_properties = pd.DataFrame(properties, columns=['area_px', 'perimeter_px'])
    df_dist = bbox_min_dist(bboxs)
    dist_thr = df_dist <= dist_threshold
    while True:
        dist_thr_new = dist_thr.dot(dist_thr)
        if dist_thr_new.equals(dist_thr):
            break
        else:
            dist_thr = dist_thr_new
    dist_thr.drop_duplicates(inplace=True)
    set_conts = []
    set_bboxs = []
    big_bbox = []
    properties_sum = []
    for i in dist_thr.index:
        set_conts.append(np.array(conts)[dist_thr.T[i]])
        set_bboxs.append(df_bboxs[dist_thr.T[i]].values)
        big_bbox.append(get_big_bbox(df_bboxs[dist_thr.T[i]]))
        properties_sum.append(df_properties[dist_thr.T[i]].sum().tolist())
    df = pd.DataFrame()
    df['contours'] = set_conts
    df['bboxs'] = set_bboxs
    df['big_bbox'] = big_bbox
    df[['area_px', 'perimeter_px']] = properties_sum
    return df


def out_data(out, CONV_LU_INV, thr_1=10):
    df_all = []
    for un in np.unique(out):
        if un != 0:
            cont = get_contours(np.array(out == un))
            df_dmg = merge_conts_by_bbox_min_dist(cont, dist_threshold=thr_1)
            df_dmg['un'] = [un] * len(df_dmg)
            df_dmg['DistressType'] = [CONV_LU_INV[un]] * len(df_dmg)
            df_all.append(df_dmg)
    return pd.concat(df_all, ignore_index=True)


def PCI_merge_severities(out, CONV_LU_INV, engin_PCI_convert, thr_area=None, thr_1=10, thr_2=20):
    df = out_data(out, CONV_LU_INV, thr_1=thr_1)
    df[['pci_cod', 'severity']] = [engin_PCI_convert[dtyp] if dtyp in engin_PCI_convert.keys() else [0, 0] for dtyp in
                                   df.DistressType]
    # print('Following info has not been taken into account:\n',df[df.pci_cod==0])
    dTyps = df.pci_cod.unique()
    data = []
    set_conts = []
    set_bboxs = []
    big_bbox2 = []
    area_sum = []
    perimeter_sum = []
    for dtype in dTyps:
        if dtype != 0:
            # big bbox near
            df_temp = df[df.pci_cod == dtype].copy()
            df_temp.reset_index(inplace=True)
            bboxs = df_temp.big_bbox.tolist()
            df_bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
            df_dist = bbox_min_dist(bboxs)
            dist_thr = df_dist <= thr_2
            while True:
                dist_thr_new = dist_thr.dot(dist_thr)
                if dist_thr_new.equals(dist_thr):
                    break
                else:
                    dist_thr = dist_thr_new
            dist_thr.drop_duplicates(inplace=True)

            for i in dist_thr.index:
                df_g = df_temp[dist_thr.T[i]].groupby(['pci_cod', 'severity']).sum()
                # dType severity, with area > 60%, will take all the near areas with same pci_cod,
                # if damages are ~50-50, they will work independently...
                # this is a temporal solution and depends on the feedback
                if df_g.area_px.max() / df_g.area_px.sum() > 0.6:
                    data.append(list(df_g[df_g.area_px == df_g.area_px.max()].index[0]))
                    contss = []
                    for conts in df_temp[dist_thr.T[i]].contours:
                        contss += list(conts)
                    set_conts.append(contss)
                    set_bboxs.append(df_bboxs[dist_thr.T[i]].values)
                    big_bbox2.append(get_big_bbox(df_bboxs[dist_thr.T[i]]))
                    area_sum.append(df_g.area_px.sum())
                    perimeter_sum.append(df_g.perimeter_px.sum())
                else:
                    data += [list(x) for x in
                             zip(df_temp[dist_thr.T[i]].pci_cod.tolist(), df_temp[dist_thr.T[i]].severity.tolist())]
                    set_conts += df_temp[dist_thr.T[i]].contours.tolist()
                    set_bboxs += df_temp[dist_thr.T[i]].bboxs.tolist()
                    big_bbox2 += df_temp[dist_thr.T[i]].big_bbox.tolist()
                    area_sum += df_temp[dist_thr.T[i]].area_px.tolist()
                    perimeter_sum += df_temp[dist_thr.T[i]].perimeter_px.tolist()
    df = pd.DataFrame(data, columns=['pci_cod', 'severity'])
    df['contours'] = set_conts
    df['bboxs'] = set_bboxs
    df['big_bbox'] = big_bbox2
    df['area_px'] = area_sum
    df['perimeter_px'] = perimeter_sum

    if thr_area:
        return df[df.area_px >= thr_area].reset_index(drop=True)
    else:
        return df