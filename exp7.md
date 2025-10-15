#samplecode:
# Generate empty target sequence of length 1 (start token)
target_seq = np.zeros((1, 1), dtype='int32')
# We don’t have an explicit “start” token in this toy example,
# so we can just leave target_seq[0,0] = 0 (padding index) or some default.

stop_condition = False
decoded_tags = []
while not stop_condition:
    output_tokens, h, c = decoder_model_inf.predict([target_seq] + states_value)
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_tag = idx2tag.get(sampled_token_index, None)
    decoded_tags.append(sampled_tag)

    # Exit condition
    if sampled_tag is None or len(decoded_tags) >= max_decoder_seq_length:
        stop_condition = True

    # Update the target sequence (length 1)
    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = sampled_token_index

    # Update states
    states_value = [h, c]

return decoded_tags
#output"<img width="342" height="169" alt="498708753-3e2c4ff1-cdce-4d94-b34b-86339f62a29b (1)" src="https://github.com/user-attachments/assets/caae3af9-ec09-4833-a2e1-8e5fbe7613d4" />
