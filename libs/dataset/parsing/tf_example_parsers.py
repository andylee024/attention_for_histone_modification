import tensorflow as tf

# TODO: figure out a cleaner way to handle these constants 
SEQUENCE_SHAPE = (1000, 4)
ANNOTATION_SHAPE = (75, 320)

def parse_attention_example(tf_example):
    """Parse tensorflow example type specifically assumed to be attention type.
    
    :param tf_example: example type from tensorflow
    :return: dictionary of tensors with the following attributes.
        
        'sequence'      : sequence tensor (1000, 4)
        'label'         : label tensor (919,)
        'annotation'    : annotation tensor (75, 320)
    """

    # specify features in attention example  
    features_map = {
        'sequence_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'annotation_raw': tf.FixedLenFeature([], tf.string)}

    # parse tf example for internal tensors
    parsed_example = tf.parse_single_example(tf_example, features_map)

    # decode examples
    sequence_raw = tf.decode_raw(parsed_example['sequence_raw'], tf.uint8)
    label_raw = tf.decode_raw(parsed_example['label_raw'], tf.uint8)
    annotation_raw = tf.decode_raw(parsed_example['annotation_raw'], tf.float32)

    # parsed tensors are flat so reshape if needed
    # cast to floats for attention task
    sequence = tf.cast(tf.reshape(sequence_raw, SEQUENCE_SHAPE), dtype=tf.float32)
    label = tf.cast(label_raw, dtype=tf.float32)
    annotation = tf.reshape(annotation_raw, ANNOTATION_SHAPE)

    return {'sequence': sequence, 'label': label, 'annotation': annotation}

