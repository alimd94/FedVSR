# this code doesn't needs to be run in the main training script but it needs sudo privilege to access CPU energy consumption.
# as we already provided training code we just added this code as a guide to where to put them in the main training script.

import pyRAPL
import time

# fit function in the main training script with logging code added
def fit(self, parameters,config):
    set_parameters(self.net, parameters)
    # meter = pyRAPL.Measurement('energy_test')

    # with meter:
    pyRAPL.setup()
    meter = pyRAPL.Measurement('bar')
    meter.begin()
    start_time = time.time()
    loss = train(self.net, self.trainloader,1,self.partition_id)
    end_time = time.time()
    total_time = end_time - start_time
    meter.end()
    log_file = f"metrics_fl_name.txt"
    with open(log_file, "a") as f:
        f.write(f"duration train-fl_name: {total_time:.6f}")
        f.write("\n")
        f.write("list of the CPU energy consumption train-fl_name: {}".format(sum([_/1000000 for _ in meter.result.pkg])))
        f.write("\n")
        f.write("CPU usage wattage train-fl_name: {}".format(sum([_/1000000 for _ in meter.result.pkg])/(meter.result.duration/1000000)))
        f.write("\n")
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        f.write("\n")

    return get_parameters(self.net), len(self.trainloader), {'loss': loss}, 



# aggregate_fit function in the main server script with logging code added
def aggregate_fit(self,server_round,results,failures):

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        if self.round == 0:
            adaptive_step = 1.0
        else:
            global TOTAL_RND
            adaptive_step = 1.0 - (self.round / TOTAL_RND)


        self.round += 1

        pyRAPL.setup()
        meter = pyRAPL.Measurement('bar')
        meter.begin()
        start_time = time.time()
        weights_avg, mixing_coeff, hellinger_dist = aggregate_with_hellinger_distance(results,weight_results,adaptive_step)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Mixing Coefficient: {mixing_coeff}, Hellinger dist: {hellinger_dist}")
        meter.end()
        log_file = f"metrics_fl_name.txt"
        with open(log_file, "a") as f:
            f.write(f"duration aggregate-fl_name: {total_time:.6f}")
            f.write("\n")
            f.write("list of the CPU energy consumption aggregate-fl_name: {}".format(sum([_/1000000 for _ in meter.result.pkg])))
            f.write("\n")
            f.write("CPU usage wattage aggregate-fl_name: {}".format(sum([_/1000000 for _ in meter.result.pkg])/(meter.result.duration/1000000)))
            f.write("\n")
            f.write("------------------------------------------------------------------")
            f.write("\n")

        weights_avg = ndarrays_to_parameters(weights_avg)
        
        glb_dir = self.base_work_dir
        os.makedirs(glb_dir, exist_ok=True)
        
        if weights_avg is not None and server_round % 1 == 0 and adaptive_step > 0.0:
            if type(weights_avg) == type([]):
                save_model_weights(server_round, weights_avg, glb_dir)
            else:
                weights_avg: List[np.ndarray] = flwr.common.parameters_to_ndarrays(weights_avg)
                save_model_weights(server_round, weights_avg, glb_dir)
            
            weights_avg = ndarrays_to_parameters(weights_avg)

        global RND
        RND += 1

        return weights_avg, {}