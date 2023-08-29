# -*- encoding: utf-8 -*-
#-------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
#-------------------------------------------------#

from flask import Flask
from flask import request
from flask import Response
from flask import stream_with_context

from cvStreamer.streamer import Yolo5Streamer

app = Flask( __name__ )
streamer = Yolo5Streamer()
src = 0

isStreaming = False

try:
    streamer.run(src)
    isStreaming = True
except GeneratorExit:
    print( '[wandlab]', 'disconnected stream' )
    streamer.stop()

@app.route('/stream')
def stream():

    try :
        
        return Response(
                                stream_with_context( stream_gen() ),
                                mimetype='multipart/x-mixed-replace; boundary=frame' )
        
    except Exception as e :
        
        print('[wandlab] ', 'stream error : ',str(e))

def stream_gen():
    if isStreaming:
        while True :
            frame = streamer.bytescode()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')