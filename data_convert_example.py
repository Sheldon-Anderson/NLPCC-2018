"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys
import codecs

import tensorflow as tf
from tensorflow.core.example import example_pb2

reload(sys)
sys.setdefaultencoding("utf-8")

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')

def _binary_to_text():
  reader = codecs.open(FLAGS.in_file, 'rb')
  writer = codecs.open(FLAGS.out_file, 'w', "utf-8")
  cnt = 1
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      # print("key: ", type(key))
      # print("value: ", type(tf_example.features.feature[key].bytes_list.value[0]))
      examples.append('%s=%s' % (key.encode("utf-8"), tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
    print "cnt: ", cnt
    cnt += 1
  reader.close()
  writer.close()
  print "binary to text finished!"


def _text_to_binary():
  inputs = codecs.open(FLAGS.in_file, 'r', "utf-8").readlines()
  writer = codecs.open(FLAGS.out_file, 'wb')
  cnt = 1
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      pos = feature.index('=', 0)
      (k, v) = (feature[:pos], feature[pos + 1:])
      # print("k: ", type(k))
      # print("v: ", type(v))
      tf_example.features.feature[k.encode('utf-8')].bytes_list.value.extend([v.encode("utf-8")])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
    print "cnt: ", cnt
    cnt += 1
  writer.close()
  print "text to binary finished!"


def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()


if __name__ == '__main__':
  tf.app.run()
