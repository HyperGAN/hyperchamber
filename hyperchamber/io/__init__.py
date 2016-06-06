import requests
import json

import sys

from json import JSONEncoder

class HCEncoder(JSONEncoder):
  def default(self, o):
    if(hasattr(o, '__call__')): # is function
      return "function:" +o.__module__+"."+o.__name__
    else:
      return o.__dict__    

def get_api_path(end):
  #return "http://localhost:3000/api/v1/"+end
  return "https://hyperchamber.255bits.com/api/v1/"+end

def apikey(apikey):
  """Sets up your api key."""
  print("TODO: Api keys")

def sample(config, samples):
  """Upload a series of samples.  Each sample has keys 'image' and 'label'. 
  Images are ignored if the rate limit is hit."""
  url = get_api_path('intrinsic.json')
  multiple_files = []
  images = [s['image'] for s in samples]
  labels = [s['label'] for s in samples]
  for image in images:
    multiple_files.append(('images', (image, open(image, 'rb'), 'image/png')))
  headers = {"config": json.dumps(config, cls=HCEncoder), "labels": json.dumps(labels)}
  try:
      r = requests.post(url, files=multiple_files, headers=headers, timeout=30)
      return r.text
  except:
      e = sys.exc_info()[0]
      print("Error while calling hyperchamber - ", e)
      return None

def record(config, result, max_retries=10):
  """Records results on hyperchamber.io.  Used when you are done testing a config."""
  url = get_api_path('run.json')
  data = {'config': config, 'result': result}
  retries = 0
  while(retries < max_retries):
      try:
          r = requests.post(url, data=json.dumps(data, cls=HCEncoder), headers={'Content-Type': 'application/json'}, timeout=30)
          return r.text
      except:
          e = sys.exc_info()[0]
          print("Error while calling hyperchamber - retrying ", e)
          retries += 1

