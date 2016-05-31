import requests

def get_api_path(end):
  return "http://localhost:3000/api/v1/"+end

def apikey(apikey):
  """Sets up your api key."""
  print("TODO: Api keys")

def sample(config, images):
  """Upload a series of images.  Images are ignored if the rate limit is hit."""
  url = get_api_path('intrinsic.json')
  multiple_files = []
  for image in images:
    multiple_files.append(('images', (image, open(image, 'rb'), 'image/png')))
  r = requests.post(url, json=config, files=multiple_files, timeout=5)
  return r.text

def record(config, result):
  """Records results on hyperchamber.io.  Used when you are done testing a config."""
  url = get_api_path('run.json')
  data = {'config': config, 'result': result}
  r = requests.post(url, json=data, timeout=30)
  return r.text
