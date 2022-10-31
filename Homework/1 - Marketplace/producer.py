"""
This module represents the Producer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Class that represents a producer.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        self.producer_id = self.marketplace.register_producer()
        i = 0
        while True:
            product = self.products[i][0]
            qty = self.products[i][1]
            wait_time = self.products[i][2]
            for _ in range(qty):
                while True:
                    published = self.marketplace.publish(self.producer_id, product)
                    if not published:
                        sleep(self.republish_wait_time)
                    else:
                        sleep(wait_time)
                        break
            i = (i + 1) % len(self.products)
