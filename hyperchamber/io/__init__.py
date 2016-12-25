import requests
import json
import sys
import os
import uuid

from json import JSONEncoder

class MissingHCKeyException(Exception):
    pass

# for function serialization
class HCEncoder(JSONEncoder):
  def default(self, o):
    if(hasattr(o, '__call__')): # is function
      return "function:" +o.__module__+"."+o.__name__
    else:
      try:
          return o.__dict__    
      except AttributeError:
          try:
             return str(o)
          except AttributeError:
              return super(o)


def get_api_path(end):
  return "https://hyperchamber.255bits.com/api/v1/"+end

def get_headers(no_content_type=False):
  if("HC_API_KEY" not in os.environ):
    raise MissingHCKeyException("hyperchamber.io api key needed.  export HC_API_KEY='...'");
  apikey = os.environ["HC_API_KEY"]
  if(no_content_type):
      return {
        'apikey': apikey
      }
  return {
    'Content-Type': 'application/json',
    'apikey': apikey
  }

def sample(config, samples):
  """Upload a series of samples.  Each sample has keys 'image' and 'label'. 
  Images are ignored if the rate limit is hit."""
  url = get_api_path('sample.json')
  multiple_files = []
  images = [s['image'] for s in samples]
  labels = [s['label'] for s in samples]
  for image in images:
    multiple_files.append(('images', (image, open(image, 'rb'), 'image/png')))
  headers=get_headers(no_content_type=True)
  headers["config"]= json.dumps(config, cls=HCEncoder)
  headers["labels"]= json.dumps(labels)
  print("With headers", headers)

  try:
      r = requests.post(url, files=multiple_files, headers=headers, timeout=30)
      return r.text
  except requests.exceptions.RequestException:
      e = sys.exc_info()[0]
      print("Error while calling hyperchamber - ", e)
      return None

def measure(config, result, max_retries=10):
  """Records results on hyperchamber.io.  Used when you are done testing a config."""
  url = get_api_path('measurement.json')
  data = {'config': config, 'result': result}
  retries = 0
  while(retries < max_retries):
      try:
          r = requests.post(url, data=json.dumps(data, cls=HCEncoder), headers=get_headers(), timeout=30)
          return r.text
      except requests.exceptions.RequestException:
          e = sys.exc_info()[0]
          print("Error while calling hyperchamber - retrying ", e)
          retries += 1


def load_config(id):
    url = get_api_path('config/'+id+'.json')
    r = requests.get(url, headers=get_headers())
    config = json.loads(r.text)
    if(not config):
        config = {}
    config['parent_uuid']=id
    config["uuid"]=uuid.uuid4().hex
    return config
