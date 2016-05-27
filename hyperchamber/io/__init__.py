import requests

def get_api_path(end):
  return "http://localhost:3000/api/v1/"+end

def apikey(apikey):
  """Sets up your api key."""
  print("TODO: Api keys")

def model(model):
  """Sets up the model you would like to use."""
  print("Model", model)

def sample(config, images):
  """Upload a series of images.  Images are ignored if the rate limit is hit."""
  print("sample")
  url = get_api_path('intrinsic.json')
  print(url)
  multiple_files = []
  for image in images:
    multiple_files.append(('images', (image, open(image, 'rb'), 'image/png')))
  r = requests.post(url, files=multiple_files, timeout=5)
  print("response", r)
  return r.text

def record(config, result):
  """Records results on hyperchamber.io.  Used when you are done testing a config."""
  print("record")
  url = get_api_path('run.json')
  data = {'config': config, 'result': result}
  r = requests.post(url, data=data, timeout=30)
  print("response", r)
  return r.text
