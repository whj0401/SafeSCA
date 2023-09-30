# -*- coding: utf-8 -*-

client_envs = {
    "RENDEZVOUS": "env://",
    "WORLD_SIZE": "2",
    "RANK": "1",
    "MASTER_ADDR": "XXX.XXX.XXX.XXX",
    "MASTER_PORT": "12345",
    "BACKEND": "gloo"
}

server_envs = {
    "RENDEZVOUS": "env://",
    "WORLD_SIZE": "2",
    "RANK": "0",
    "MASTER_ADDR": "XXX.XXX.XXX.XXX",
    "MASTER_PORT": "12345",
    "BACKEND": "gloo"
}
