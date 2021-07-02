from collections import OrderedDict

class LandmarksMapping300wWFLW:
    face_parts_indices_300w = {
        "jaw" : list(range(0, 17)),
        "left_eyebrow" : list(range(17,22)),
        "right_eyebrow" : list(range(22,27)),
        "nose" : list(range(27,36)),
        "left_eye" : list(range(36, 42)),
        "right_eye" : list(range(42, 48)),
        "left_eye_poly": list(range(36, 42)),
        "right_eye_poly": list(range(42, 48)),
        "mouth" : list(range(48,68)),
        "eyes" : list(range(36, 42))+list(range(42, 48)),
        "eyebrows" : list(range(17,22))+list(range(22,27)),
        "eyes_and_eyebrows" : list(range(17,22))+list(range(22,27))+list(range(36, 42))+list(range(42, 48)),
    }

    face_parts_indices_wflw = {
        "jaw" : list(range(0,33)),
        "left_eyebrow" : list(range(33,42)),
        "right_eyebrow" : list(range(42,51)),
        "nose" : list(range(51, 60)),
        "left_eye" : list(range(60, 68))+[96],
        "right_eye" : list(range(68, 76))+[97],
        "left_eye_poly": list(range(60, 68)),
        "right_eye_poly": list(range(68, 76)),
        "mouth" : list(range(76, 96)),
        "eyes" : list(range(60, 68))+[96]+list(range(68, 76))+[97],
        "eyebrows" : list(range(33,42))+list(range(42,51)),
        "eyes_and_eyebrows" : list(range(33,42))+list(range(42,51))+list(range(60, 68))+[96]+list(range(68, 76))+[97],
    }

    index_map_300w_to_wflw = OrderedDict()
    index_map_300w_to_wflw.update(dict(zip(range(0,17),range(0,34,2)))) # челюст | 17 точки
    index_map_300w_to_wflw.update(dict(zip(range(17,22),range(33,38)))) # горен ръб на лявата вежда | 5 точки
    index_map_300w_to_wflw.update(dict(zip(range(22,27),range(42,47)))) # горен ръб на дясната вежда | 5 точки
    index_map_300w_to_wflw.update(dict(zip(range(27,36),range(51,60)))) # нос | 9 точки
    index_map_300w_to_wflw.update({36:60}) # ляво око | 6 точки
    index_map_300w_to_wflw.update({37:61})
    index_map_300w_to_wflw.update({38:63})
    index_map_300w_to_wflw.update({39:64})
    index_map_300w_to_wflw.update({40:65})
    index_map_300w_to_wflw.update({41:67})
    index_map_300w_to_wflw.update({42:68}) # дясно око | 6 точки
    index_map_300w_to_wflw.update({43:69})
    index_map_300w_to_wflw.update({44:71})
    index_map_300w_to_wflw.update({45:72})
    index_map_300w_to_wflw.update({46:73})
    index_map_300w_to_wflw.update({47:75})
    index_map_300w_to_wflw.update(dict(zip(range(48,68),range(76,96)))) # уста | 20 точки

    index_map_wflw_to_300w = OrderedDict()
    index_map_wflw_to_300w.update({ i : None for i in range(0, 98) })
    index_map_wflw_to_300w.update({ v : k for k,v in index_map_300w_to_wflw.items() })

def index_300w_to_wflw(idx):
    return LandmarksMapping300wWFLW.index_map_300w_to_wflw[idx]

def index_wflw_to_300w(idx):
    return LandmarksMapping300wWFLW.index_map_wflw_to_300w[idx]

def list_300w_to_wflw(landmarks):
    result = [ (None, None) for i in range(0, 98) ]
    for k,v in LandmarksMapping300wWFLW.index_map_300w_to_wflw.items():
        result[v] = landmarks[k]
    return result

def list_wflw_to_300w(landmarks):
    result = []
    for k,v in LandmarksMapping300wWFLW.index_map_wflw_to_300w.items():
        if v is not None:
            result.append(landmarks[k])
    return result

def dict_300w_to_wflw(landmarks_dict):
    result = {}
    for part,landmarks in landmarks_dict.items():
        result[part] = [ (None, None) for i in range(len(LandmarksMapping300wWFLW.face_parts_indices_wflw[part])) ]
        for i in range(len(landmarks)):
            idx_300w = LandmarksMapping300wWFLW.face_parts_indices_300w[part][i]
            idx_wflw = LandmarksMapping300wWFLW.index_map_300w_to_wflw[idx_300w]
            idx_wflw_part = LandmarksMapping300wWFLW.face_parts_indices_wflw[part].index(idx_wflw)
            result[part][idx_wflw_part] = landmarks[i]
    return result

def dict_wflw_to_300w(landmarks_dict):
    result = {}
    for part,landmarks in landmarks_dict.items():
        result[part] = [ (None, None) for i in range(len(LandmarksMapping300wWFLW.face_parts_indices_300w[part])) ]
        for i in range(len(landmarks)):
            idx_wflw = LandmarksMapping300wWFLW.face_parts_indices_wflw[part][i]
            idx_300w = LandmarksMapping300wWFLW.index_map_wflw_to_300w[idx_wflw]
            if idx_300w is not None:
                idx_300w_part = LandmarksMapping300wWFLW.face_parts_indices_300w[part].index(idx_300w)
                result[part][idx_300w_part] = landmarks[i]
    return result
