# Imports
import datetime
import discord
from discord import Activity, ActivityType
import json
import nltk
import numpy as np
import os
import pickle
import psutil
import pyttsx3
import random
import requests
import subprocess
import sympy
import sys
from datetime import date
from discord.ext import commands
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from better_profanity import profanity

# Configuration
start_time = datetime.datetime.now()
mod_roles = ['Gearsmith', 'Admin', 'Administrator', 'Moderator', 'Mod', 'Keycard']
with open('./config.json') as f:
    data = json.load(f)
    prefixes = data["Prefix"]
    token = data["Token"]


neural_status = Activity(
    type=ActivityType.watching,
    name="neural activity 🧠"
)

neural_waiting_status = Activity(
    type=ActivityType.watching,
    name="for commands"
)

post_idle_status = Activity(
    type=ActivityType.watching,
    name="for commands"
)

# Define intent
intents = discord.Intents.all()
intents.message_content = True
intents.members = True

# Create prefix
def get_prefix(bot, message):
    for prefix in prefixes:
        if message.content.startswith(prefix):
            return prefix
    # Default prefix if none of the defined prefixes match
    return "a/"
client = commands.Bot(command_prefix=get_prefix, intents=intents)

# Event handler for when the bot is ready
@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.online)
    print('AI has connected in as {0.user}'.format(client))

## Neural segment -- This is where the AI magic happens

# Downloads any updates.
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("src/neural_data/training-data.json").read())
words = pickle.load(open('src/neural_data/words.pkl', 'rb'))
classes = pickle.load(open('src/neural_data/classes.pkl', 'rb'))
model = load_model('src/neural_data/echo.h5')

# Tokenize the words in the sentence and lemmatize them
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert the sentence to a bag of words representation
def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Use the trained model to predict the class of the input sentence
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get a random response from the list of responses associated with the predicted class
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# bot, marked 'client' as it's ran client-side for the end-user
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # Handles DMs
    if isinstance(message.channel, discord.DMChannel) or message.guild is None:
        await client.change_presence(status=discord.Status.online)
        responses = ["I don't reply to direct messages. Sorry.",
                    'Could you tell me directly on a server I operate on please?',
                    "Sorry I didn't quite catch that, tell me directly by tagging me in a server.",
                    "I appreciate the reply but please ask me directly on a server.",
                    "Tag me with your prompts. in a server, and I'll be happy to help if I can.",
                    "For a more accurate response, please mention me directly in your question in a server.",
                    "I'm here to assist, on a server... Feel free to tag me and ask your question if I can help I will!.",
                    "Direct questions are the best way to get a response from me, if you're on a server.",
                    "If you tag me in your message, I'll provide a direct answer, I don't do replies in the DMs."]
        response = random.choice(responses)
        await message.channel.send(response)
        await client.change_presence(status=discord.Status.idle, activity=post_idle_status)
        await handle_dm_message(message)
        return
    # Handles in reference (replies)
    if message.reference:
        # Check if the message is a reply
        replied_message = await message.channel.fetch_message(message.reference.message_id)
        if replied_message.author == client.user:
            await client.change_presence(status=discord.Status.online)
            # The user is replying to the bot
            # You can process this reply here
            responses = ["I don't reply to direct responses anymore.",
                        'Could you tell me directly please?',
                        "Sorry I didn't quite catch that, tell me directly by tagging me.",
                        "I appreciate the reply but please ask me directly.",
                        "Tag me with your prompts, and I'll be happy to help if I can.",
                        "For a more accurate response, please mention me directly in your question.",
                        "I'm here to assist. Feel free to tag me and ask your question if I can help I will!.",
                        "Direct questions are the best way to get a response from me.",
                        "If you tag me in your message, I'll provide a direct answer, I don't do replies."]
            await message.channel.send(random.choice(responses))
            await client.change_presence(status=discord.Status.idle, activity=post_idle_status)
            return
    # Checks for profanity, all the time based on message content against the .txt filter, not context!
    if profanity.contains_profanity(message.content):
            await client.change_presence(status=discord.Status.online)
            await message.delete()
            responses = ['Oh, my circuits are buzzing! Infraction detected, message removed.',
                        'Ta-da... Message vanished like magic!',
                        'Impeccable timing, as always. Message removed and onto Dyno!',
                        "You've been caught in the act! Logged with Dyno.",
                        'Huzzah! Potential infraction neutralized, message to Dyno!',
                        'My database is chock-full of your shenanigans. Message logged and removed.',
                        "Engaging moderator mode. Poof, your message is gone!",
                        'Our databases are on fire with this one. Message removed for safety!',
                        "Safety first! Your message has been whisked away." ,
                        'Spick and span! Message removed to keep things clean.',
                        'Time to zip it! Infraction detected and message removed.',
                        'Whoosh! Your message has vanished into the digital abyss!',
                        'Astra-nomical! Message removed, just like that.',
                        "Logged, sealed, and delivered! Your message is no more.",
                        'Swish and flick! Message removed like a wizard spell!',
                        'Message detected, message removed! Poof!',
                        'No room for mischief here! Message has been vanished.',
                        'Zap! Your message has vanished from the realm of chat!',
                        'My, oh my! Message removed for safety reasons.',
                        'Better safe than sorry! Your message is gone.',
                        'Message flagged and removed, just in the nick of time!',
                        'Astra-cadabra. Message gone, like magic!',
                        'Caught in the act! Message detected and removed.',
                        "My sensors don’t miss a thing! Message removed.",
                        'Infraction alert! Your message has been promptly removed.',
                        'I blink, I detect, and I remove! Message has vanished.']
            await message.channel.send(random.choice(responses))
            await client.change_presence(status=discord.Status.idle, activity=post_idle_status)
            username = str(message.author).split('#')[0]
            user_message = str(message.content)
            channel = str(message.channel.name)
            print(f'{username}: {user_message} ({channel})')

            await client.process_commands(message)
            # detected_prefix = get_prefix(client, message)
            # print(f'Prefix: {detected_prefix}')

    # Check if the bot is mentioned in the message
    if client.user.mentioned_in(message):
        await client.change_presence(status=discord.Status.dnd, activity=neural_status)
        # Use the AI to generate a response
        res = predict_class(message.content)
        if res:  # Check if res is not empty, as sometimes it can be!
            response = get_response(res, intents)
        else:
            responses = ["Oh, I don't know how to answer that right now..."]
            response = random.choice(responses)

        await message.channel.send(response)
        await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

## Moderation

# Mimic
@client.slash_command(name="mimic", description="Send a mimic message to a channel.")
@commands.has_any_role(*mod_roles)
async def mimic_slash(ctx, channel: discord.TextChannel, *, message: str):
    # Check if the user has the 'send_messages' permission in the specified channel
    if not channel.permissions_for(ctx.author).view_channel:
        await ctx.respond("You do not have permission to view that channel.", ephemeral=True)
        return
    if channel.permissions_for(ctx.author).send_messages:
        await channel.send(f"{message}")
        await ctx.respond("Anonymous message sent successfully.", ephemeral=True)
    else:
        await ctx.respond("You do not have permission to send messages to that channel.", ephemeral=True)

# Quick mod
@client.slash_command(name="moderate", description="Kick, time out, or ban a user!")
@commands.has_any_role(*mod_roles)
async def moderate_slash(ctx, target: discord.Member, action: str):
    # Check if the action is one of 'kick', 'ban', or 'timeout'
    if action.lower() == 'timeout':
        await target.add_roles(discord.utils.get(ctx.guild.roles, name="Timeout"))
        await ctx.respond(f"Timed out {target.mention}.", ephemeral=True)
    elif action.lower() == 'kick':
        await target.kick()
        await ctx.respond(f"Kicked {target.mention}.", ephemeral=True)
    elif action.lower() == 'ban':
        await target.ban()
        await ctx.respond(f"Banned {target.mention}.", ephemeral=True)
    else:
        await ctx.respond("Invalid action. Please use 'kick', 'ban', or 'timeout'.", ephemeral=True)

# Delete
@client.slash_command(name="delete", description="Delete a specified number of messages.")
@commands.has_any_role(*mod_roles)
async def delete_slash(ctx, num_messages: int):
    # Check if the number of messages to delete is within a reasonable range
    if 1 <= num_messages <= 100:
        # Delete the specified number of messages, including the command itself
        await ctx.channel.purge(limit=num_messages + 0)
        await ctx.respond(f"Deleted {num_messages} messages.", ephemeral=True)
    else:
        await ctx.respond("Please specify a number of messages to delete between 1 and 100.", ephemeral=True)

# Echo
@client.slash_command(name="echo", description="Like Mimic, but Echo the provided input.")
@commands.has_any_role(*mod_roles)
async def echo_slash(ctx, *, echo):
    await client.change_presence(status=discord.Status.online)
    await ctx.respond(echo)
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

## MISC

# 8Ball
@client.slash_command(name="8ball", description="Ask the magic 8-ball a question.")
async def eightball_slash(ctx, *, question):
    await client.change_presence(status=discord.Status.online)
    EIGHT_BALL_RESPONSES = ["It is certain.",
                            "It is decidedly so.",
                            "Without a doubt.",
                            "Yes, definitely.",
                            "You may rely on it.",
                            "As I see it, yes.",
                            "Most likely.",
                            "Outlook good.",
                            "Yes.",
                            "Signs point to yes.",
                            "Reply hazy, try again.",
                            "Ask again later.",
                            "Better not tell you now.",
                            "Cannot predict now.",
                            "Concentrate and ask again.",
                            "Don't count on it.",
                            "My reply is no.",
                            "My sources say no.",
                            "Outlook not so good.",
                            "Very doubtful."]
    response = random.choice(EIGHT_BALL_RESPONSES)
    await ctx.respond(f"**Question:** {question}\n**8-Ball Response:** {response}")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Status
@client.slash_command(name="status", description="Is the bot is online?")
async def first_slash(ctx):
    await client.change_presence(status=discord.Status.online)
    await ctx.respond("I'm online!")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Roll
@client.slash_command(name="roll", description="Roll a dice, 1d6 specifically.")
async def roll(ctx):
    await client.change_presence(status=discord.Status.online)
    import random
    result = random.randint(1, 6)
    await ctx.respond(f"You rolled a {result}!")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Coinflip
@client.slash_command(name="coinflip", description="Flip a coin.")
async def coinflip(ctx):
    await client.change_presence(status=discord.Status.online)
    import random
    result = random.choice(["Heads", "Tails"])
    await ctx.respond(f"The coin landed on {result}.")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Catfact
@client.slash_command(name="catfact", description="Get a random cat fact.")
async def catfact(ctx):
    await client.change_presence(status=discord.Status.online)
    response = requests.get("https://catfact.ninja/fact")
    data = response.json()
    fact = data["fact"]
    await ctx.respond(f"Cat Fact: {fact}")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Joke
@client.slash_command(name="joke", description="Tell a random joke, it's a short list.")
async def joke(ctx):
    await client.change_presence(status=discord.Status.online)
    jokes = ["Why did the chicken cross the road? To get to the other side!", "Why don't scientists trust atoms? Because they make up everything!"]
    joke = random.choice(jokes)
    await ctx.respond(joke)
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Uptime
@client.slash_command(name="uptime", description="Check the bot's uptime.")
async def uptime_slash(ctx):
    await client.change_presence(status=discord.Status.online)
    current_time = datetime.datetime.now()
    uptime = current_time - start_time
    uptime_str = str(uptime).split('.')[0]  # Convert the timedelta to a string and removes microseconds!
    await ctx.respond(f"Bot Uptime: {uptime_str}")
    await client.change_presence(status=discord.Status.idle, activity=post_idle_status)

# Polls
active_polls = {}
@client.slash_command(name="poll", description="Create a poll.")
async def poll_slash(ctx, question: str, options_str: str):
    # Split the options string into a list of options
    options = options_str.split(",")
    # Defines the options.
    if len(options) < 2 or len(options) > 10:
        await ctx.respond("A poll must have 2 to 10 options split by a comma (,).", ephemeral=True)
        return
    # Create an embed for the poll
    embed = discord.Embed(title="Poll", description=question, color=0xcd853f)
    for i, option in enumerate(options):
        embed.add_field(name=f"Option {i+1}", value=option.strip(), inline=False)
    # Send the poll message and add reactions
    poll_message = await ctx.send(embed=embed)
    for i in range(len(options)):
        await poll_message.add_reaction(f"{i+1}\N{COMBINING ENCLOSING KEYCAP}")
    # Store the poll for reference and cleanup
    active_polls[poll_message.id] = {
        "question": question,
        "options": options,
        "author_id": ctx.author.id,
    }
@client.event
async def on_raw_reaction_add(payload):
    if payload.message_id in active_polls:
        poll_data = active_polls[payload.message_id]
        if payload.user_id != poll_data["author_id"]:
            # If someone voted.
            option_number = int(payload.emoji.name[0]) - 1
            option = poll_data["options"][option_number]
            await payload.member.send(f"You voted for '{option}' in the poll: {poll_data['question']}")

# Calculator
@client.slash_command(name="calculate", description="Perform basic calculations.")
async def calculate_slash(ctx, expression: str):
    try:
        result = sympy.sympify(expression)
        await ctx.respond(f"Result: {result}", ephemeral=True)
    except Exception as e:
        await ctx.respond(f"An error occurred: {str(e)}", ephemeral=True)

# Anon messaging
@client.slash_command(name="anonymous", description="Send an anonymous message to a channel.")
async def anonymous_slash(ctx, channel: discord.TextChannel, *, message: str):
    # Check if the user has the 'send_messages' permission in the specified channel!
    if not channel.permissions_for(ctx.author).view_channel:
        await ctx.respond("You do not have permission to view that channel.", ephemeral=True)
        return
    if channel.permissions_for(ctx.author).send_messages:
        await channel.send(f"**Anonymous**: {message}")
        await ctx.respond("Anonymous message sent successfully.", ephemeral=True)
    else:
        await ctx.respond("You do not have permission to send messages to that channel.", ephemeral=True)

## Cogs - DO NOT REMOVE OR MODIFY UNLESS YOU KNOW WHAT YOU'RE DOING

# Load Cogs
for filename in os.listdir('cogs'):
    if filename.endswith('.py'):
        client.load_extension(f'cogs.{filename[:-3]}')

@commands.slash_command(name="cogs", description="List all active cogs.")
async def list_cogs_slash(ctx):
    cogs = client.cogs
    cog_names = [cog for cog in cogs]

    if cog_names:
        cog_list = "\n".join(cog_names)
        await ctx.respond(f"**Active Cogs:**\n{cog_list}", ephemeral=True)
    else:
        await ctx.respond("No active cogs found.", ephemeral=True)

@client.command()
async def load(ctx, extension):
    client.load_extension(f'cogs.{extension}')

@client.command()
async def unload(ctx, extension):
    client.unload_extension(f'cogs.{extension}')

# Runs the bot
client.run(token)