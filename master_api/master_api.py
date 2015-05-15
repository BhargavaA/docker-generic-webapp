
from flask import Flask, request
import json

app = Flask(__name__)

import os
MASTERREDIS_HOST = os.environ.get('MASTERREDIS_PORT_6379_TCP_ADDR', 'localhost')
MASTERREDIS_PORT = int(os.environ.get('MASTERREDIS_PORT_6379_TCP_PORT', 6379))

import redis
r = redis.StrictRedis(host=MASTERREDIS_HOST, port=MASTERREDIS_PORT)

@app.route('/', methods=["POST","OPTIONS","GET"])
def hello():

    num_times_site_visited = r.incr('num_times_site_visited') 
    # args = request.get_json()    

    return_str = "Hello World! This site has been visited %d times" % num_times_site_visited

    return return_str

import time
@app.route('/testconn', methods=["POST","OPTIONS","GET"])
def test_conn():

    args = request.get_json()    

    time_sleep = args.get('sleep_time',0.05)
    time.sleep(time_sleep)
    args['server_compute_time'] = time_sleep 

    return json.dumps(args), 200, {'Access-Control-Allow-Origin':'*', 'Content-Type':'application/json'}
    

# Run this flask application
if __name__ == '__main__':
   app.run(debug=True)




