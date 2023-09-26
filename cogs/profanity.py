from asyncio.streams import _ClientConnectedCallback
import discord
from discord.ext import commands
import json
import os
from better_profanity import profanity

# Profanity filter list.
profanity.load_censor_words_from_file("cogs/profanity.txt")

# Config.
class Initial(commands.Cog):
    def __init__ (self, client):
        self.client = _ClientConnectedCallback

def setup(client):
    client.add_cog(Initial(client))