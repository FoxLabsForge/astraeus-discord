from discord.ext import commands
import discord

class PingCog(commands.Cog):
    def __init__(self, client):
        self.client = client

    @commands.slash_command(name="ping", description="Check the bot's ping.", ephemeral=True)
    async def ping_slash(self, ctx):
        await self.client.change_presence(status=discord.Status.online)
        bot_latency = round(self.client.latency * 1000)
        await ctx.respond(f"Ping: {bot_latency}ms")
        await self.client.change_presence(status=discord.Status.idle)

def setup(client):
    client.add_cog(PingCog(client))