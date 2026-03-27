import requests


def get_embedding(text: str) -> list[float]:
    try:
        # 构建请求体（对应 Java 的 Map）
        request_json = {
            "inputs": [text]
        }

        embed_url = "http://127.0.0.1:8901/embed"

        # 发送 POST 请求
        response = requests.post(
            embed_url,
            json=request_json,  # 自动序列化为 JSON
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            raise RuntimeError(f"向量请求失败: {response.text}")

        # 解析返回 JSON
        outer_array = response.json()

        if not outer_array:
            return []

        inner_array = outer_array[0]  # 取第一条向量

        # 转 float list（Python 一般是 float64）
        embedding = [float(x) for x in inner_array]

        return embedding

    except Exception as e:
        raise RuntimeError("无法获取向量数据") from e

def build_embeddings(data):
    embedding = []
    for datum in data:
        embedding.append(get_embedding(datum))
    return embedding
