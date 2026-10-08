"""Regression: media-form requests must preserve the required OpenAI model field."""
import argparse
import json
import urllib.error
import urllib.request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:42667")
    args = parser.parse_args()
    for fields in ({}, {"model": ""}, {"model": 123}):
        body = {"input": {"content": [{"type": "text", "text": "hello"}]}, **fields}
        request = urllib.request.Request(args.url + "/v1/embeddings", json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
            raise AssertionError("invalid OpenAI model accepted for media-form input")
        except urllib.error.HTTPError as error:
            result = json.load(error)
            assert error.code == 400 and result["error"]["param"] == "model", result
    print("OpenAI media model validation: passed")
