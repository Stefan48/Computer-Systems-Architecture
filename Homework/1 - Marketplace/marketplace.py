"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Lock

class Marketplace:
    """
    Class that represents the Marketplace. It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """
    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue associated with each producer
        """
        # dictionary for all available products: key = product, value = [(producer_id, quantity)]
        self.available = {}
        self.queue_size_per_producer = queue_size_per_producer
        self.queue_size = []
        self.max_producer_id = -1
        self.max_cart_id = -1
        self.carts = []
        self.lock_register = Lock()
        self.lock_new_cart = Lock()
        self.lock_available = Lock()
        self.lock_queue_size = Lock()

    def register_producer(self):
        """
        Returns an id for the producer that calls this.
        """
        with self.lock_register:
            self.max_producer_id += 1
            self.queue_size.append(0)
        return self.max_producer_id

    def publish(self, producer_id, product):
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False, it should wait and then try again.
        """
        with self.lock_queue_size:
            if self.queue_size[producer_id] == self.queue_size_per_producer:
                return False
        with self.lock_available:
            if product in self.available:
                producer_in_list = False
                for i in range(len(self.available[product])):
                    if self.available[product][i][0] == producer_id:
                        producer_in_list = True
                        self.available[product][i] = \
						    (self.available[product][i][0], self.available[product][i][1] + 1)
                        break
                if not producer_in_list:
                    self.available[product].append((producer_id, 1))
            else:
                self.available[product] = [(producer_id, 1)]
        with self.lock_queue_size:
            self.queue_size[producer_id] += 1
        return True

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """
        with self.lock_new_cart:
            self.max_cart_id += 1
            self.carts.append([])
        return self.max_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False, it should wait and then try again
        """
        available = False
        with self.lock_available:
            if product in self.available:
                producer = self.available[product][0][0]
                self.available[product][0] = (producer, self.available[product][0][1] - 1)
                if self.available[product][0][1] == 0:
                    del self.available[product][0]
                    if not self.available[product]:
                        del self.available[product]
                available = True
        if available:
            self.carts[cart_id].append((producer, product))
            return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """
        for i in range(len(self.carts[cart_id])):
            if self.carts[cart_id][i][1] == product:
                producer = self.carts[cart_id][i][0]
                del self.carts[cart_id][i]
                # first, decrement producer's queue size
                # since calling 'publish' is going to increase it
                with self.lock_queue_size:
                    self.queue_size[producer] -= 1
                self.publish(producer, product)
                break

    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        products = []
        for i in range(len(self.carts[cart_id])):
            producer = self.carts[cart_id][i][0]
            products.append(self.carts[cart_id][i][1])
            with self.lock_queue_size:
                self.queue_size[producer] -= 1
        self.carts[cart_id] = []
        return products
