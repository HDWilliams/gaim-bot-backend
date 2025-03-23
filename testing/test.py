import requests
headers = {
    'content-type': 'application/json',
    'x-api-key': 'bgA9f635kGGMRMwhFKQk2iSGxAoYedq9sPH4p22a'
}
url = 'https://2b0su814ae.execute-api.us-east-1.amazonaws.com/gaim-bot-dev-1/download_faiss_index'

body = {
    'prompt': 'What is eldin ring?'
}

try:
    response = requests.post(url, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f'Error {e}. Message: {e.response.json()}')