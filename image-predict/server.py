from bottle import route, run, request, response
from predictor import predictor
import json

worker = predictor.ImagePredictor()
worker.load_model()

@route('/predict', method='POST')
def predict():
    if request.json == None:
        response.code = 500
        return {'code': 1, 'msg': 'no request data'}
    try:
        uri = request.json['data']['uri']
    except Exception:
        response.code = 500
        return {'code': 2, 'msg': 'request data is invalid'}
    ret = worker.predict(uri)
    return {'code': 0, 'results': ret}

if __name__ == '__main__':
    run(port='8080', debug=True)