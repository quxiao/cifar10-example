from bottle import route, run, request, response
from predictor import predictor
import json

worker = predictor.ImagePredictor()
worker.load_model()

@route('/v1/image/classify/predict', method='POST')
def predict():
    if request.json == None:
        response.code = 500
        return {'code': 1, 'msg': 'no request data'}
    try:
        uri = request.json['data']['uri']
    except Exception:
        response.code = 500
        return {'code': 2, 'msg': 'request data is invalid'}
    try:
        ret = worker.predict(uri)
    except Exception:
        response.code = 500
        return {'code': 3, 'msg': 'server error'}
    return {'code': 0, 'results': ret}

if __name__ == '__main__':
    run(host='0.0.0.0', port='8080', debug=True)