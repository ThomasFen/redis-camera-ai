import cv2
import redisAI
import numpy as np
from redisgears import executeCommand as execute
from PIL import Image, ImageDraw
import io
import base64
import json 
from time import time

# Globals for downsampling
_mspf = 1000 / 10.0      # Msecs per frame (initialized with 10.0 FPS)
_next_ts = 0             # Next timestamp to sample a frame
class SimpleMovingAverage(object):
    # Taken from https://github.com/RedisGears/EdgeRealtimeVideoAnalytics
    def __init__(self, value=0.0, count=7):
        '''
        @value - the initialization value
        @count - the count of samples to keep
        '''
        self.count = int(count)
        self.current = float(value)
        self.samples = [self.current] * self.count

    def __str__(self):
        return str(round(self.current, 3))

    def add(self, value):
        '''
        Adds the next value to the average
        '''
        v = float(value)
        self.samples.insert(0, v)

        o = self.samples.pop()
        self.current = self.current + (v-o)/self.count
class Profiler(object):
    # Taken from https://github.com/RedisGears/EdgeRealtimeVideoAnalytics
    names = []
    data = {}
    last = None

    def __init__(self):
        pass

    def __str__(self):
        s = ''
        for name in self.names:
            s = '{}{}:{}, '.format(s, name, self.data[name])

        return(s[:-2])

    def __delta(self):
        '''
        Returns the time delta between invocations in milliseconds
        '''
        now = time()*1000
        if self.last is None:
            self.last = now

        value = now - self.last
        self.last = now

        return value

    def start(self):
        '''
        Starts the profiler
        '''
        self.last = time()*1000
        return self

    def add(self, name):
        '''
        Adds/updates a step's duration
        '''
        value = self.__delta()

        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def assign(self, name, value):
        '''
        Assigns a step with a value
        '''
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def get(self, name):
        '''
        Gets a step's value
        '''
        return self.data[name].current

def downsampleStream(x):
    ''' Drops input frames to match FPS '''
    global _mspf, _next_ts
    ts, _ = map(int, str(x['id']).split('-'))         # Extract the timestamp part from the message ID
    sample_it = _next_ts <= ts
    if sample_it:                                           # Drop frames until the next timestamp is in the present/past
        _next_ts = ts + _mspf
    return sample_it
    
def processImage(img, height):
    '''
    Resize a rectangular image to a padded square (letterbox)
    '''
    color = (127.5, 127.5, 127.5)
    shape = img.shape[:2]

    ratio = float(height) / max(shape)
    newShape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    dw = (height - newShape[0]) / 2
    dh = (height - newShape[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.resize(img, newShape, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=color)
    img = np.asarray(img, dtype=np.float32)

    '''
    Normalize 0..255 to 0..1.00
    '''
    img /= 255.0

    return img


def process_cropped_image(img):
    # A grayscale image is required.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Width and height have to be 224.
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32)
    # Normalize
    img=img/255
    # Input dimensions wll be (<Nr. of faces>,224,224,1)
    img = np.expand_dims(img, axis=-1)

    return img

def index_to_emotion(index):
    emotions =  {0:
                'anger',
                1:
                'contempt',
                2:
                'disgust',
                3:
                'fear',
                4:
                'happy',
                5:
                'neutral',
                6:
                'sad',
                7:
             'surprise'    }

    return emotions[index]
    
async def getFaces(x):
    prf = Profiler().start()
    '''
    Read the image from the stream's message.
    '''
    buf = io.BytesIO(x['value']['img'])
    pilImage = Image.open(buf)
    numpyImage = np.array(pilImage)
    prf.add('loading-image')

    """ Prepare the image for blazeface. """
    imageSize = 256 # Front model would be 128 in size.
    img_np = processImage(numpyImage, imageSize)
    prf.add('preprocess-blazeface')

    """ Runs blazeface and its script as an directed acyclic graph (DAG) on an input image from the stream """
    img_ba = bytearray(img_np.tobytes())
    image_tensor = redisAI.createTensorFromBlob('FLOAT', [1, 256, 256, 3], img_ba)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.TensorSet('img', image_tensor)
    DAGRunner.ModelRun(name='blazeface:model', inputs=['img'], outputs=['reg', 'class'])
    DAGRunner.ScriptRun(name='blazeface:script', func='boxes', inputs=['reg','class'], outputs=['bounding_boxes'])
    DAGRunner.TensorGet('bounding_boxes')
    scriptReply = await DAGRunner.Run()
    prf.add('blazeface')

    '''
    The script outputs bounding boxes
    '''
    scriptOutput = scriptReply[0]
    shape = redisAI.tensorGetDims(scriptOutput)
    scriptData = redisAI.tensorGetDataAsBlob(scriptOutput)
    scriptBoxes = np.frombuffer(scriptData, dtype=np.float32).reshape(shape)

    boxes = []
    wholePadding = max(pilImage.width, pilImage.height) - min(pilImage.width, pilImage.height)
    sidePadding = wholePadding / 2
    isWide = pilImage.width == max(pilImage.width, pilImage.height)

    '''
    Iterate boxes 
    '''
    for box in scriptBoxes:
        '''
        Remove zero-confidence detections 
        '''
        if box[-1] == 0.0:
            continue
        '''
        Descale bounding box coordinates back to original image size
        ''' 
        if isWide:
            y1 = (pilImage.height + wholePadding) * box[0] - sidePadding
            y2 = (pilImage.height + wholePadding) * box[2] - sidePadding
            x1 = pilImage.width * box[1]
            x2 = pilImage.width * box[3]

        else:
            x1 = (pilImage.height + wholePadding) * box[1] - sidePadding
            x2 = (pilImage.height + wholePadding) * box[3] - sidePadding
            y1 = pilImage.height * box[0]
            y2 = pilImage.height * box[2]   

        rescaledBox = (x1, y1, x2, y2)
        boxes.append(rescaledBox)
    prf.add('rescale-boxes')

    return x['key'], x['id'], pilImage, boxes, x['value']['userId'], x['value']['conferenceId'], prf

async def getEmotions(x): 
    key, id, pilImage, boxes, userId, conferenceId, prf = x

    """ Extract the faces from the frame and convert them to the input format required for the emotion model."""
    faces =  []
    for box in boxes:
        face = pilImage.crop(box)
        face = np.asarray(face)
        face_np = process_cropped_image(face)
        faces.append(face_np)
    prf.add('preprocess-emotion')

    """ Transform the faces into an input tensor for the emotion model. """
    faces = np.asarray(faces)
    imgages_ba = bytearray(faces.tobytes())
    faces_count = len(faces)      
    images_tensor = redisAI.createTensorFromBlob('FLOAT', [faces_count, 224, 224, 1], imgages_ba)

    """ Run the model to detect emotions. """
    modelRunner = redisAI.createModelRunner('emotion')
    redisAI.modelRunnerAddInput(modelRunner, 'input_emotion', images_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output')
    emotion_model_reply =  await redisAI.modelRunnerRunAsync(modelRunner)
    prf.add('emotion')

    return key, id, pilImage, boxes, emotion_model_reply, userId, conferenceId, prf

def storeResults(x):
    key, id, pilImage, boxes, emotion_model_reply, userId, conferenceId, prf = x
    global _mspf

    """ The model output is converted to a JSON suitable format and the most probable emotion is determined for every face. """
    emotions_per_face_dicts = []
    all_emotion_values = redisAI.tensorToFlatList(emotion_model_reply[0])
    emotions_per_face = [all_emotion_values[i:i + 8] for i in range(0, len(all_emotion_values), 8)]

    for faceIndx, emotions in enumerate(emotions_per_face):
        emotions_per_face_dicts.append({'raw':{}, 'dominantEmotion':{}})
        # Find name of most likely emotion.
        dominant = index_to_emotion(emotions.index(max(emotions)))
        # Add the name for the emotion that was determined to be predominant.
        emotions_per_face_dicts[faceIndx]['dominantEmotion'] = dominant 
        # Add the numeric value of every emotion type. They lie between 0 and 1.
        for i, emotion in enumerate(emotions):
            emotion_type = index_to_emotion(i)
            emotions_per_face_dicts[faceIndx]['raw'][emotion_type] = emotion

    """ Store model outputs, userId and id of the corresponding input stream entry in an output stream. The current time is returned."""
    emotions_formatted = {'userId': userId,'emotions': emotions_per_face_dicts}
    result_id = execute('XADD', '{}:results'.format(key), 'MAXLEN', '~', 1000, '*','ref', id, 'userId', userId,'conferenceId', conferenceId,'boxes', boxes, 'emotions', json.dumps(emotions_formatted))
    result_ms = int(str(result_id).split('-')[0])
    inputStreamMsec = int(str(id).split('-')[0])
    prf.assign('total',result_ms - inputStreamMsec)

    """ Store the output values of the emotion model in time series."""
    for faceIndx, emotions_of_face in enumerate(emotions_per_face_dicts):
        for emotion_type, emotion_value in emotions_of_face['raw'].items():
            execute('TS.ADD', f'{conferenceId}:{userId}:{faceIndx}:{emotion_type}', inputStreamMsec, emotion_value, 'LABELS', 'conferenceId', conferenceId, 'userId', userId)

    """ Record profiler steps """
    labels = ['LABELS', 'stream', key]
    for name in prf.names:
        execute('TS.ADD', '{}:prf_{}'.format(key, name), inputStreamMsec,
                prf.data[name].current, *labels, 'data', name)

    """ Drawing is ignored because it's eventually going to be removed from the gear for performance reasons."""
    avg_duration = prf.get('total')
    _mspf = avg_duration * 1.05  # A little extra leg room  

    return key, id, pilImage, boxes, emotions_per_face_dicts

def drawResults(x):
    key, id, pilImage, boxes, emotions_per_face_dicts = x

    """ Draw bounding boxes around faces on the frame. """
    draw = ImageDraw.Draw(pilImage)
    for box in boxes:
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), width=1, outline='red')

    """ Add emotiontype to boxes """   
    numpyImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_BGR2RGB)
    for indx, box in enumerate(boxes):
        cv2.putText(numpyImage, '{}'.format(emotions_per_face_dicts[indx]['dominantEmotion']), (int(box[0]), int(box[1])),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

    """ Add stream key and entry id to the frame."""
    cv2.putText(numpyImage, '{}:{}'.format(key, id), (10, pilImage.height - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
    _, img = cv2.imencode('.jpg', numpyImage)

    """ Store the modified frame. """
    execute('XADD', '{}:draw'.format(key), 'MAXLEN', '~',
                        1000, '*', 'ref', id, 'img', base64.b64encode(img.tobytes()))
                        


gb = GearsBuilder('StreamReader')
gb.filter(downsampleStream) 
gb.map(getFaces)  
gb.filter(lambda x: isinstance(x[3], list) and  len(x[3]) > 0) # Stop when no faces detected.
gb.map(getEmotions)
gb.map(storeResults)
gb.map(drawResults)
gb.register('main')


# gb = GearsBuilder('StreamReader', defaultArg='camera:1')
# gb.map(runBlazeface)  
# gb.run(fromId='0-0')




