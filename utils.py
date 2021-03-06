import glob
import cv2
import numpy as np
import sqlite3

def load_brakelights(s, rgb=True):
    global X_mean
    '''
    Two classes: brakelight or reject
    Each class has it's own folder
    '''
    all_files = []

    # folders must have the same names as the labels that are defined
    for l in s.labels:
        all_files.append(glob.glob("./brakelights/%s/*.png" % (l)))

    X_train = None
    X_test = None
    if rgb:
        X_train = np.empty((0,3,s.img_size, s.img_size),np.float32)# contains data
        X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    else:
        X_train = np.empty((0,1,s.img_size, s.img_size),np.float32)# contains data
        X_test = np.empty((0,1,s.img_size, s.img_size),np.float32)# contains data

    y_train = np.empty((0), np.int32)# contains labels    
    y_test = np.empty((0), np.int32)# contains labels  
    
    for i,filelist in enumerate(all_files):
        # first, create array out of all training data
        cnt = 0
        for j,f in enumerate(filelist):
            cnt += 1
            img = None
            if rgb:
                img = cv2.resize(cv2.imread(f) / 255., ( s.img_size, s.img_size))
                img = img.transpose(2,0,1).reshape(3,  s.img_size,  s.img_size)
            else:
                img = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) / 255., (s.img_size, s.img_size))

            if rgb:
                X_train = np.append(X_train, np.array([img.astype(np.float32)]), axis=0)
            else:
                X_train = np.append(X_train, np.array([[img.astype(np.float32)]]), axis=0)
            y_train = np.append(y_train, i)
        print "Got %d files for label %s" % (cnt, s.labels[i])


    # now, shuffle
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm) # create a permutation, then use this for Data and Labels
    X_train = X_train[perm]
    y_train = y_train[perm]

    # now split
    X_test = X_train[0.8*X_train.shape[0]:] 
    y_test = y_train[0.8*y_train.shape[0]:]
    X_train = X_train[:0.8*X_train.shape[0]]
    y_train = y_train[:0.8*y_train.shape[0]]

    # save to disk for later
    np.save("X_train", X_train)
    np.save("X_test", X_test)
    np.save("y_train", y_train.astype(np.int32))
    np.save("y_test", y_test.astype(np.int32))
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32)

def load_road_pictures(s):
    '''
    use the database of Tobi to receive annotated images according to their highwaytype
    '''
    dbPath = "/home/shared/data/TobisGpsSequence/sequences_960_720_manual.db"
    sequenceDB = sqlite3.connect(dbPath)
    db = sequenceDB.cursor()

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"primary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")
    primaryframes = db.fetchall()[:1000]

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"secondary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")

    secondaryframes = db.fetchall()[:1000]

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"tertiary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")

    tertiaryframes = db.fetchall()[:1000]
    
    print "Got %d primary, %d secondary ,%d tertiary highwayframes" % (len(primaryframes), len(secondaryframes), len(tertiaryframes))

    X_train = np.empty((0,3,s.img_size, s.img_size),np.float32)# contains data
    y_train = np.empty((0), np.int32)# contains labels    
    X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    y_test = np.empty((0), np.int32)# contains labels  

    for index,files in enumerate([primaryframes, secondaryframes, tertiaryframes]):
        cnt_train = 0
        cnt_test = 0

        for i,f in enumerate(primaryframes):
            img = cv2.resize(cv2.imread(f[0]+f[1])/255., (s.img_size,s.img_size))
            img = img.transpose(2,0,1).reshape(3, s.img_size, s.img_size)
            if i < len(primaryframes) * 0.8:
                cnt_train += 1
                X_train = np.append(X_train, np.array([img.astype(np.float32)]), axis=0)
                y_train = np.append(y_train, index)
            else:
                cnt_test += 1
                X_test = np.append(X_test, np.array([img.astype(np.float32)]), axis=0)
                y_test = np.append(y_test, index)   

        print "Got %d train and %d test files for label %s" % (cnt_train, cnt_test, s.labels[index])
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32)



def load_faces(s, rgb=True):
    global X_mean
    '''
    we have three classes: Tobi, Mariam, Reject
    Each class has it's own folder
    '''
    all_files = []

    # folders must have the same names as the labels that are defined
    for l in s.labels:
        all_files.append(glob.glob("./faces_120/%s/*.png" % (l)))

    X_train = None
    X_test = None
    if rgb:
        X_train = np.empty((0,3,s.img_size, s.img_size),np.float32)# contains data
        X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    else:
        X_train = np.empty((0,1,s.img_size, s.img_size),np.float32)# contains data
        X_test = np.empty((0,1,s.img_size, s.img_size),np.float32)# contains data

    y_train = np.empty((0), np.int32)# contains labels    
    y_test = np.empty((0), np.int32)# contains labels  
    
    for i,filelist in enumerate(all_files):
        # first, create array out of all training data
        cnt = 0
        for j,f in enumerate(filelist):
            cnt += 1
            img = None
            if rgb:
                img = cv2.resize(cv2.imread(f) / 255., ( s.img_size, s.img_size))
                img = img.transpose(2,0,1).reshape(3,  s.img_size,  s.img_size)
            else:
                img = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) / 255., (s.img_size, s.img_size))

            if rgb:
                X_train = np.append(X_train, np.array([img.astype(np.float32)]), axis=0)
            else:
                X_train = np.append(X_train, np.array([[img.astype(np.float32)]]), axis=0)
            y_train = np.append(y_train, i)
        print "Got %d files for label %s" % (cnt, s.labels[i])


    # now, shuffle
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm) # create a permutation, then use this for Data and Labels
    X_train = X_train[perm]
    y_train = y_train[perm]

    # now split
    X_test = X_train[0.8*X_train.shape[0]:] 
    y_test = y_train[0.8*y_train.shape[0]:]
    X_train = X_train[:0.8*X_train.shape[0]]
    y_train = y_train[:0.8*y_train.shape[0]]

    # save to disk for later
    np.save("X_train", X_train)
    np.save("X_test", X_test)
    np.save("y_train", y_train.astype(np.int32))
    np.save("y_test", y_test.astype(np.int32))
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32)

def load_road_pictures(s):
    '''
    use the database of Tobi to receive annotated images according to their highwaytype
    '''
    dbPath = "/home/shared/data/TobisGpsSequence/sequences_960_720_manual.db"
    sequenceDB = sqlite3.connect(dbPath)
    db = sequenceDB.cursor()

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"primary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")
    primaryframes = db.fetchall()[:1000]

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"secondary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")

    secondaryframes = db.fetchall()[:1000]

    db.execute("select folder,img_uri from frames,sequences where context_highwaytype=\"tertiary\" and frames.id_sequence = sequences.id and (sequences.id=40 or sequences.id=43 or sequences.id=47 or sequences.id=49 or sequences.id=51 or sequences.id=63 or sequences.id=48) order by RANDOM()")

    tertiaryframes = db.fetchall()[:1000]
    
    print "Got %d primary, %d secondary ,%d tertiary highwayframes" % (len(primaryframes), len(secondaryframes), len(tertiaryframes))

    X_train = np.empty((0,3,s.img_size, s.img_size),np.float32)# contains data
    y_train = np.empty((0), np.int32)# contains labels    
    X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    y_test = np.empty((0), np.int32)# contains labels  

    for index,files in enumerate([primaryframes, secondaryframes, tertiaryframes]):
        cnt_train = 0
        cnt_test = 0

        for i,f in enumerate(primaryframes):
            img = cv2.resize(cv2.imread(f[0]+f[1])/255., (s.img_size,s.img_size))
            img = img.transpose(2,0,1).reshape(3, s.img_size, s.img_size)
            if i < len(primaryframes) * 0.8:
                cnt_train += 1
                X_train = np.append(X_train, np.array([img.astype(np.float32)]), axis=0)
                y_train = np.append(y_train, index)
            else:
                cnt_test += 1
                X_test = np.append(X_test, np.array([img.astype(np.float32)]), axis=0)
                y_test = np.append(y_test, index)   

        print "Got %d train and %d test files for label %s" % (cnt_train, cnt_test, s.labels[index])
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32)


