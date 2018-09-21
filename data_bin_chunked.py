import codecs
import os
import struct
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('in_dir', '', 'dir to input *.bin file')
tf.app.flags.DEFINE_string('out_dir', '', 'dir to output chunked *.bin file')

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = os.path.join(FLAGS.in_dir, '%s.bin') % set_name
  reader = codecs.open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(FLAGS.out_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with codecs.open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(FLAGS.out_dir):
    os.mkdir(FLAGS.out_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print "Splitting %s data into chunks..." % set_name
    chunk_file(set_name)
  print "Saved chunked data in %s" % FLAGS.out_dir


if __name__ == "__main__":
    assert FLAGS.in_dir and FLAGS.out_dir
    chunk_all()
