import json
import logging
import threading

from flask import Flask
from flask import Response
from flask import request

from svd_predictor_als import Predictor

logging.basicConfig(filename='dataset/file.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
log = logging.getLogger('App')

app = Flask(__name__)
predictor = Predictor()
predictor.train()
job_id_mapping = json.load(open("dataset/job_id_mapping.json"))
nodes = []
queries = []
num_of_invocations = {}  # json.load(open("dataset/num_of_invocations.json"))


def on_exit():
    with open("dataset/job_id_mapping.json", "w+") as file:
        json.dump(job_id_mapping, file, indent=4)
        file.close()
    with open("dataset/num_of_invocations.json", "w+") as file:
        json.dump(num_of_invocations, file, indent=4)
        file.close()
    #
    # with open("dataset/nodes", "w+") as file:
    #     json.dump(nodes, file, indent=4)
    #     file.close()
    # with open("dataset/queries", "w+") as file:
    #     json.dump(queries, file, indent=4)
    #     file.close()

    log.info("records saved successfully")


# atexit.register(on_exit)


def action_param_to_job_id(action, params):
    job_id = None
    # remove irrelevant data from params
    for irr_key in ['extractedMetadata', 'startTimes', 'commTimes']:
        if irr_key in params:
            del params[irr_key]
    match_obj = {"action": action, "params": params}
    for id in job_id_mapping:
        if job_id_mapping[id] == match_obj:
            job_id = id
    if job_id is None:
        cur_ids = [int(x) for x in job_id_mapping.keys()]
        job_id = '01' if len(cur_ids) == 0 else str(max(cur_ids) + 1).zfill(2)
        job_id_mapping[job_id] = match_obj
        # with open("dataset/job_id_mapping.json", "w+") as file:
        #     json.dump(job_id_mapping, file, indent=4)
        #     file.close()
    return job_id


def get_num_invocation(job_id):
    if job_id in num_of_invocations:
        return num_of_invocations[job_id]
    else:
        invocation = 0
        num_of_invocations[job_id] = invocation
        # with open("dataset/num_of_invocations.json", "w+") as file:
        #     json.dump(num_of_invocations, file, indent=4)
        #     file.close()
        return invocation


def inc_num_invocation(job_id):
    if job_id in num_of_invocations:
        invocation = num_of_invocations[job_id] + 1
    else:
        invocation = 1
    num_of_invocations[job_id] = invocation
    # with open("dataset/num_of_invocations.json", "w+") as file:
    #     json.dump(num_of_invocations, file, indent=4)
    #     file.close()


@app.route('/node', methods=["POST"])
def node():
    received_obj = json.loads(request.data.decode())
    log.info(f"Received Request Data: {received_obj}")
    nodes.append(received_obj)
    global predictor
    params = received_obj['parameters']
    job_name = received_obj['job_name']
    job_id = action_param_to_job_id(job_name, params)
    inc_num_invocation(job_id)
    config_id = received_obj['config_id']

    scaled_rt = float(received_obj['runtime'].split()[0])
    log.info(f"job_id: {job_id}, config_id: {config_id}, runtime: {scaled_rt}")

    predictor.add_data_node([int(job_id), config_id, scaled_rt])
    predictor.train()

    response = Response(response=None, status=204)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.route('/query', methods=["POST"])
def query():
    received_obj = json.loads(request.data.decode())
    log.info(f"Received Request Data: {received_obj}")

    job_id = action_param_to_job_id(received_obj['job_name'], received_obj['parameters'])
    invocation = get_num_invocation(job_id)
    log.info(f"job_id: {job_id}, invocation: {invocation}")
    if invocation == 0:
        response_obj = {"type": "data",
                        "config_id": str(22).zfill(2)}
    elif invocation == 1:
        response_obj = {"type": "data",
                        "config_id": str(7).zfill(2)}
    else:
        rec_config = predictor.recommend_config_for_job(int(job_id), "cost")
        response_obj = {"type": "data",
                        "config_id": str(rec_config).zfill(2)}
    log.info(f"Response: {response_obj}")
    # pprint.pprint(response_obj)
    received_obj['response'] = response_obj
    queries.append(received_obj)
    response = Response(response=json.dumps(response_obj), status=200)  # , mimetype="application/json"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.route('/save', methods=["GET", "POST"])
def save():
    # predictor.save_est_matrix()
    on_exit()
    response = Response(response='{"message": "est_matrix saved"}', status=200)  # , mimetype="application/json"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.route('/hello', methods=["GET"])
def hello():
    log.info("Hello Received")
    response = Response(response='{"message": "hello from predictor !!!"}', status=200)  # , mimetype="application/json"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
