# 图片分类推理 API

## API

### 推理服务

请求包

```
POST /v1/image/classify/predict
Content-Type: application/json

{
    "data": {
        "uri": <uri>
    }
}
```

说明

`<uri>` 可以为以下形式：

* HTTP， 网络资源，形如：http://host/path、https://host/path
* Data，Data URI Scheme形态的二进制文件，形如：data:application/octet-stream;base64,xxx。ps: 当前只支持前缀为data:application/octet-stream;base64,的数据


返回包

```
200 OK
Content-Type: application/json

{
    "code": 0,
    "msg": <string>,            // 错误信息
    "results": [
        {
            "prob": <float>,
            "label: <string>
        },{
            ...
        }
    ]
}
```