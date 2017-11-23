from data import *
import math
LARGE_HEIGHT, SMALL_HEIGHT, CHANNEL = 32, 16, 3
version = 'test1'
tf.app.flags.DEFINE_string('train_dir', '/tmp/C_GAN',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    
class C_GAN:
    
    def __init__(self, batch_size, num_epochs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
    def discriminator(self,img, reuse = False):
        f1, f2, f3 = 32, 64, 128
        with tf.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()            
            conv1 = tf.layers.conv2d(img,f1,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
            act1 = lrelu(conv1, n='act1')
            conv2 = tf.layers.conv2d(act1,f2,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn2')
            act2 = lrelu(conv2, n='act2')
            conv3 = tf.layers.conv2d(act2,f3,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn3')
            act3 = lrelu(conv3, n='act3')
            dim = int(np.prod(act3.get_shape()[1:]))
            fc1 = tf.reshape(act3, shape=[-1, dim], name = 'fcl')
            w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
            return logits
        
    def generator(self,img, reuse = False):
        with tf.variable_scope('gen') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.layers.conv2d_transpose(img, 128, kernel_size=[5, 5], strides=[1, 1], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
            bn1= tf.contrib.layers.batch_norm(conv1, epsilon=1e-5, decay = 0.9,  updates_collections=None,  scope='bn1')
            act1 = tf.nn.relu(bn1, name='act1')
            conv2 = tf.layers.conv2d_transpose(act1, 64, kernel_size=[5, 5], strides=[1, 1], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            bn2 = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = tf.nn.relu(bn2, name='act2')
            conv3 = tf.layers.conv2d_transpose(act2, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
            act3 = tf.nn.relu(conv3, name='act3')   
            return act3
    def train(self):
        with tf.variable_scope('input'):
            #real and fake image placholders
            large_image = tf.placeholder(tf.float32, shape = [None, LARGE_HEIGHT, LARGE_HEIGHT, CHANNEL], name='large_image')
            small_image = tf.placeholder(tf.float32, shape = [None, SMALL_HEIGHT, SMALL_HEIGHT, CHANNEL], name='small_image')
            
            is_train = tf.placeholder(tf.bool, name='is_train')

        if 1:
            fake_image = self.generator(small_image)
            tf.summary.image("generated images", fake_image, max_outputs=3)

            real_result = self.discriminator(large_image)
            fake_result = self.discriminator(fake_image, reuse=True)
            
            d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(real_result, tf.ones_like(real_result)))
            d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(fake_result, tf.zeros_like(fake_result)))
            d_loss = d_loss_real + d_loss_fake
            g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(fake_result, tf.ones_like(fake_result)))


            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)

            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'dis' in var.name]
            g_vars = [var for var in t_vars if 'gen' in var.name]
            trainer_d = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=d_vars)
            trainer_g = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=g_vars)
           
            d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

            batch_num = 500
        
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # continue training
        save_path = saver.save(sess, "/tmp/model.ckpt")
        ckpt = tf.train.latest_checkpoint('./model/' + version)
        saver.restore(sess, save_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/train',sess.graph)
        
        print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, self.num_epochs))
        print('start training...')
        for i in range(self.num_epochs):
            print('\n\nEpoch: {}\n\n'.format(i))
            for j in range(batch_num):
                larges_images = get_batch('original')
                small_images = get_batch('resized')
                sess.run(d_clip)
                _, dLoss,summary = sess.run([trainer_d, d_loss,merged],
                                    feed_dict={large_image:larges_images,
                                    small_image: small_images, is_train: True})
                _, gLoss= sess.run([trainer_g, g_loss],
                                   feed_dict={small_image: small_images, is_train: True})
                print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
                train_writer.add_summary(summary,j)
                del larges_images, small_images
            """if i%500 == 0:
                if not os.path.exists('./model/' + version):
                    os.makedirs('./model/' + version)
                saver.save(sess, './model/' +version + '/' + str(i))"""
        coord.request_stop()
        coord.join(threads)

gan = C_GAN(100,500)
gan.train()
