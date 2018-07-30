#using this function, we can get batches. Feed With matrices of weights and labels
import math
def batches(batch_size, features, labels):
	
		
	assert len(features)==len(labels)
	output=[] # each element of the array output is [ batch of features , batch of labels ]
	for start_i in range(0,len(features), batch_size):
		end_i=start_i+batch_size
		batch=[features[start_i:end_i], labels[start_i:end_i]]
		output.append(batch)
	return output


