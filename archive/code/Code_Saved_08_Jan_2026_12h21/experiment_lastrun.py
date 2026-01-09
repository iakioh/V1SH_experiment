#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Januar 07, 2026, at 11:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2025.1.1')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from GetExperimentalInfo
import yaml
from numpy.random import default_rng
import math
import pandas as pd
import itertools
import atexit
import shutil
from datetime import datetime
import questplus
import json
import pickle
import random

# Parameters loaded / calculated from file
ExpInfo_path = os.path.join(os.path.dirname(__file__), '../ExpInfo.yaml')
with open(ExpInfo_path, 'r') as f:
    ExpInfo = yaml.safe_load(f)

StimInfo = ExpInfo["Stimulus"]

nTrials_test = ExpInfo["Procedure&Conditions"]["NumberOfTrials"]
nTrials_train = ExpInfo["Procedure&Conditions"]["NumberOfPracticeTrials"]
track_time = ExpInfo["Procedure&Conditions"]["MaxTime_min"] is not None
if track_time:
    MAX_TIME_s = 60 * ExpInfo["Procedure&Conditions"]["MaxTime_min"] # s
    practice_start_time_s = None

    
seed = ExpInfo["Procedure&Conditions"]["RandomSeed"]
if seed is not None:
    raise Warning("Should randomize over subjects and sessions.")
else:
    now = datetime.now()
    # times of seconds since Jan 1, 1970 UTC
    # * 1000000 to get microsecond variability
    # modulo 2**31 to avoid overflow
    # int32 since required type of seed for numpy
    seed = np.int32(int(datetime.now().timestamp() * 1000000) % (2**31))  # e.g., 20251208143052
    rng = default_rng(seed)
    ExpInfo["Procedure&Conditions"]["RandomSeed"] = int(seed)

frequency = StimInfo["GaborPatch"]["SpatialFrequency_cpd"] # cycles per degree
orientations_degrees = StimInfo["GaborPatch"]["Orientation_degrees"] # degrees
mitchell_contrast = StimInfo["GaborPatch"]["MitchellContrast"] 
sigma = StimInfo["GaborPatch"]["Sigma_degrees"] # degrees
assert StimInfo["GaborPatch"]["PatchSize_sigma"] == 6, "Gabor patch size must be 6 sigma"
spacing = StimInfo["GaborPatch"]["PatchSize_sigma"] * sigma  # degrees
jitter_amount = StimInfo["Jitter"]["Range_sigma"] * sigma  # max jitter in degrees

n_rows = StimInfo["Grid"]["Rows"]
vertical_seperation = StimInfo["VerticalSeperation_degrees"] # degrees

x_eccentricity_degrees = StimInfo["Eccentricity_degrees"]["x"] # degrees
y_eccentricity_degrees = StimInfo["Eccentricity_degrees"]["y"] # degrees

displacement_range_cols = [StimInfo["HorizontalDisplacement_cols"]["Minimum"], StimInfo["HorizontalDisplacement_cols"]["Maximum"]] # columns
displacement_step_cols = StimInfo["HorizontalDisplacement_cols"]["StepSize"] # columns
displacements_cols = np.arange(displacement_range_cols[0], displacement_range_cols[1] + displacement_step_cols, displacement_step_cols)

fixation_shape = StimInfo["Fixation"]["Shape"]
fixation_duration = StimInfo["Fixation"]["Duration_s"] # seconds
fixation_diameter = StimInfo["Fixation"]["Diameter_degrees"] # degrees
central_exclusion_zone_shape = StimInfo["CentralExclusionZone"]["Shape"]
central_exclusion_width_degrees = StimInfo["CentralExclusionZone"]["Width_degrees"] # degrees
central_exclusion_height_degrees = StimInfo["CentralExclusionZone"]["Height_degrees"] # degrees

mean_range_cols = [ExpInfo["Procedure&Conditions"]["Staircase"]["Mean_cols"]["Minimum"], ExpInfo["Procedure&Conditions"]["Staircase"]["Mean_cols"]["Maximum"]] # columns
mean_step_cols = ExpInfo["Procedure&Conditions"]["Staircase"]["Mean_cols"]["StepSize"] # columns
means_cols = np.arange(mean_range_cols[0], mean_range_cols[1] + mean_step_cols, mean_step_cols)
std_range_cols = [ExpInfo["Procedure&Conditions"]["Staircase"]["StandardDeviation_cols"]["Minimum"], ExpInfo["Procedure&Conditions"]["Staircase"]["StandardDeviation_cols"]["Maximum"]] # columns
std_step_cols = ExpInfo["Procedure&Conditions"]["Staircase"]["StandardDeviation_cols"]["StepSize"] # columns
stds_cols = np.arange(std_range_cols[0], std_range_cols[1] + std_step_cols, std_step_cols)
lapse_rates = np.array(ExpInfo["Procedure&Conditions"]["Staircase"]["LapseRate"])

STAIRCASE = None # overwrite with selected staircase each trial later
STIMULUS_DICT = None

# Ensure hardcoded parameters are correct
assert StimInfo["Fixation"]["Color"] == "white", "Fixation stimulus should be set to 'white'."
assert StimInfo["Jitter"]["Type"] == "uniform", "Spatial jitter distribution should be set to 'uniform'."
assert central_exclusion_zone_shape == "ellipse", "Central exclusion zone shape should be set to 'ellipse'."
# Run 'Before Experiment' code from SimulateResponse
from scipy.stats import norm

# Psychometrische Funktion (kumulative Normalverteilung)
def psychometric_function(x, mu=0, sigma=1):
    """
    Berechnet die Wahrscheinlichkeit einer "rechts"-Antwort
    basierend auf einer kumulativen Normalverteilung.
    
    Parameter:
    x: Stimulus-Intensität oder -Position
    mu: Mittelwert (Point of Subjective Equality, PSE)
    sigma: Standardabweichung (Schwelle)
    
    Return: Wahrscheinlichkeit zwischen 0 und 1
    """
    return norm.cdf(x, loc=mu, scale=sigma)

# Simulierte Antwort basierend auf der psychometrischen Funktion
def simulate_response(stimulus_value, mu=0, sigma=1, left_key='left', right_key='right'):
    """
    Simuliert einen Tastendruck basierend auf der psychometrischen Funktion.
    
    Parameter:
    stimulus_value: Der aktuelle Stimulus-Wert
    mu: PSE der psychometrischen Funktion
    sigma: Schwelle der psychometrischen Funktion
    left_key: Name der linken Taste
    right_key: Name der rechten Taste
    
    Return: Tastenname (left_key oder right_key)
    """
    # Wahrscheinlichkeit für "rechts"
    p_right = psychometric_function(stimulus_value, mu, sigma)
    
    # Sample aus Bernoulli-Verteilung
    response = np.random.rand() < p_right
    
    return right_key if response else left_key
# Run 'Before Experiment' code from Timer
import time
# Run 'Before Experiment' code from SimulateResponse
from scipy.stats import norm

# Psychometrische Funktion (kumulative Normalverteilung)
def psychometric_function(x, mu=0, sigma=1):
    """
    Berechnet die Wahrscheinlichkeit einer "rechts"-Antwort
    basierend auf einer kumulativen Normalverteilung.
    
    Parameter:
    x: Stimulus-Intensität oder -Position
    mu: Mittelwert (Point of Subjective Equality, PSE)
    sigma: Standardabweichung (Schwelle)
    
    Return: Wahrscheinlichkeit zwischen 0 und 1
    """
    return norm.cdf(x, loc=mu, scale=sigma)

# Simulierte Antwort basierend auf der psychometrischen Funktion
def simulate_response(stimulus_value, mu=0, sigma=1, left_key='left', right_key='right'):
    """
    Simuliert einen Tastendruck basierend auf der psychometrischen Funktion.
    
    Parameter:
    stimulus_value: Der aktuelle Stimulus-Wert
    mu: PSE der psychometrischen Funktion
    sigma: Schwelle der psychometrischen Funktion
    left_key: Name der linken Taste
    right_key: Name der rechten Taste
    
    Return: Tastenname (left_key oder right_key)
    """
    # Wahrscheinlichkeit für "rechts"
    p_right = psychometric_function(stimulus_value, mu, sigma)
    
    # Sample aus Bernoulli-Verteilung
    response = np.random.rand() < p_right
    
    return right_key if response else left_key
# Run 'Before Experiment' code from SimulateResponse
from scipy.stats import norm

# Psychometrische Funktion (kumulative Normalverteilung)
def psychometric_function(x, mu=0, sigma=1):
    """
    Berechnet die Wahrscheinlichkeit einer "rechts"-Antwort
    basierend auf einer kumulativen Normalverteilung.
    
    Parameter:
    x: Stimulus-Intensität oder -Position
    mu: Mittelwert (Point of Subjective Equality, PSE)
    sigma: Standardabweichung (Schwelle)
    
    Return: Wahrscheinlichkeit zwischen 0 und 1
    """
    return norm.cdf(x, loc=mu, scale=sigma)

# Simulierte Antwort basierend auf der psychometrischen Funktion
def simulate_response(stimulus_value, mu=0, sigma=1, left_key='left', right_key='right'):
    """
    Simuliert einen Tastendruck basierend auf der psychometrischen Funktion.
    
    Parameter:
    stimulus_value: Der aktuelle Stimulus-Wert
    mu: PSE der psychometrischen Funktion
    sigma: Schwelle der psychometrischen Funktion
    left_key: Name der linken Taste
    right_key: Name der rechten Taste
    
    Return: Tastenname (left_key oder right_key)
    """
    # Wahrscheinlichkeit für "rechts"
    p_right = psychometric_function(stimulus_value, mu, sigma)
    
    # Sample aus Bernoulli-Verteilung
    response = np.random.rand() < p_right
    
    return right_key if response else left_key
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'experiment'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'monitor_type': 'HP ProBook 450 G8 Notebook PC',
    'mean_luminance_cdpm2': 'unknown',
    'viewing_distance_cm': '60',
    'save_in': 'personal',
    'session': '01',
    'ID': '001',
    'first_name': 'Kai',
    'surname': 'Rothe',
    'age': '25',
    'gender': 'male',
    'visual_acuity_left_dpt': 'approx. -2 to -3',
    'visual_acuity_right_dpt': 'approx. -2 to -3',
    'visual_acuity_corrected': True,
    'handedness': 'unknown',
    'eye_dominance': 'unknown',
    'diseases': 'unknown',
    'subject_comments': '',
    'skip_instructions': False,
    'skip_training': False,
    'skip_testing': False,
    'monkey_mode': False,
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'../../data/%s/ID%sS%s_%s_%s_%s/ID%sS%s_%s_%s_%s' % (expInfo['save_in'], expInfo['ID'], expInfo['session'], expInfo['surname'], expInfo['first_name'], expInfo['date'], expInfo['ID'], expInfo['session'], expInfo['surname'], expInfo['first_name'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\kai.rothe\\Documents\\V1SH_experiment\\experiment\\code\\experiment_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='HP ProBook 450 G8 Notebook PC', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('AnyButton') is None:
        # initialise AnyButton
        AnyButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='AnyButton',
        )
    if deviceManager.getDevice('GetResponse') is None:
        # initialise GetResponse
        GetResponse = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='GetResponse',
        )
    if deviceManager.getDevice('RepeatOrContinueButton') is None:
        # initialise RepeatOrContinueButton
        RepeatOrContinueButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='RepeatOrContinueButton',
        )
    if deviceManager.getDevice('AnyButton_2') is None:
        # initialise AnyButton_2
        AnyButton_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='AnyButton_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # enter 'rush' mode (raise CPU priority)
    if not PILOTING:
        core.rush(enable=True)
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Preparation" ---
    # Run 'Begin Experiment' code from GetExperimentalInfo
    # Pre-compute and select eccentricities
    session_index = int(expInfo["session"]) - 1 # expInfo["session"] starts with "001"
    abs_x_eccentricity_degrees = abs(x_eccentricity_degrees) / 2 # degrees
    abs_y_eccentricity_degrees = abs(y_eccentricity_degrees) / 2 # degrees
    eccentricities_degrees = [[abs_x_eccentricity_degrees, abs_y_eccentricity_degrees], [-abs_x_eccentricity_degrees, abs_y_eccentricity_degrees]] # upper left and right side
    xy_eccentricity_degrees = eccentricities_degrees[session_index % 2]
    
    # Pre-compute timing
    def get_frames_from_seconds(duration_s):
        refresh_rate = win.getActualFrameRate() # Hz
        return round(refresh_rate * duration_s)
    
    # name with "_" at the end to avoid clashes with global variables 
    # later injected through trial condition files
    stimulus_duration_s_ = StimInfo["GaborPatch"]["Duration_s"]
    stimulus_frames_ = get_frames_from_seconds(stimulus_duration_s_)
    fixation_frames = get_frames_from_seconds(fixation_duration)
    blank_duration_s = StimInfo["Blank"]["Duration_s"]
    blank_frames = get_frames_from_seconds(blank_duration_s)
    
    # Pre-compute column number
    
    # Get monitor parameters
    mon = win.monitor
    viewing_distance = mon.getDistance()   # in cm
    screen_width_cm = mon.getWidth()       # in cm
    
    # Compare with parameters by hand (dialog box)
    assert expInfo['monitor_type'] == win.monitor.name, f"Monitor setting (monitor type) incorrect: {win.monitor.name} (true) != {expInfo['monitor_type']} (given)."
    assert float(expInfo['viewing_distance_cm']) == viewing_distance, f"Monitor setting (viewing distance) incorrect:  {win.monitor.getDistance()} (true) != {float(expInfo['viewing_distance_cm'])} (given)."
    
    # Calculate screen width in visual degrees
    # Formula: visual_angle = 2 * arctan(size_cm / (2 * distance_cm))
    screen_width_degrees = 2 * np.degrees(np.arctan(screen_width_cm / (2 * viewing_distance)))
    screen_width_cols = int(np.ceil(screen_width_degrees / spacing))
    
    # Calculate minimum number of columns to cover the screen
    # Add 5 to each side to ensure we cover the whole screen, also edges
    n_col_plus = 5
    max_displacement_cols = int(max(displacement_range_cols[1], -displacement_range_cols[0])) # to allow shifting border
    n_col = screen_width_cols + 2 * max_displacement_cols + 2 * n_col_plus 
    
    # Assure even number of columns
    if n_col % 2 == 1:
        n_col += 1
    # Run 'Begin Experiment' code from GetTrialConditionSequence
    # Create and save conditions sequence
    
    # instruction trials
    instructions_filepath = "../instructions.csv" # hardcoded here
    instruction_conditions_df = pd.read_csv(instructions_filepath) 
    instruction_conditions_df['session'] = session_index 
    instruction_conditions_df["stimulus_frames"] = instruction_conditions_df["stimulus_duration_s"].apply(get_frames_from_seconds)
    instruction_conditions_df["x_eccentricity_degrees"] = xy_eccentricity_degrees[0]
    instruction_conditions_df["y_eccentricity_degrees"] = xy_eccentricity_degrees[1]
    instruction_conditions_df["state"] = "instruct"
    instruction_conditions_df.to_csv('conditions_temp_instruct.csv', index=False)
    
    # training trials
    conditions = list(itertools.product(n_rows, displacements_cols, orientations_degrees))
    
    def sample_from(items, n_samples): 
        # sample conditions randomly
        # if n_samples < len(items): just sample randomly
        # else, select all samples (in random order) and sample the remainder
        samples = items * (n_samples // len(items)) + random.sample(items, n_samples % len(items))
        random.shuffle(samples) # in-place shuffeling
        return samples
    conditions = sample_from(conditions, nTrials_train)
    nReps_train = 1 
    
    conditions_df = pd.DataFrame(conditions, columns=['n_row', 'displacement_cols', 'orientation_degrees'])        
    conditions_df['session'] = session_index 
    conditions_df["stimulus_frames"] = stimulus_frames_
    conditions_df["stimulus_duration_s"] = stimulus_duration_s_
    conditions_df["x_eccentricity_degrees"] = xy_eccentricity_degrees[0]
    conditions_df["y_eccentricity_degrees"] = xy_eccentricity_degrees[1]
    conditions_df["state"] = "train"
    conditions_df["instruction"] = None
    conditions_df.to_csv('conditions_temp_train.csv', index=False)
    
    # testing trials: psi staircase + randomize over conditions
    conditions = list(itertools.product(n_rows, orientations_degrees))
    nReps_test = int(math.ceil(nTrials_test / len(conditions)))
    
    def get_key(n_row, orientation_degrees):
        # % 90 to combine staircase inference for symmetrically oriented textures
        return f"n_row={n_row}_orientation_degrees={orientation_degrees % 90}"
    
    staircases = {}
    for n, o in conditions:
        assert 0 <= o < 180, "Orientations must be in [0, 180) degree range."
        key = get_key(n, o)
        if key not in staircases:
            staircases[key] = questplus.qp.QuestPlus(
                stim_domain = {"intensity": displacements_cols}, # displacement min & max
                param_domain = {"mean": means_cols, "sd": stds_cols, "lapse_rate": lapse_rates},
                outcome_domain = {"response": [1, 0]},
                func = "norm_cdf_2", # assumes lapse rate == lower asymptote
                stim_scale = "linear"
            )
    conditions_df = pd.DataFrame(conditions, columns=['n_row', 'orientation_degrees'])  
    conditions_df['session'] = session_index
    conditions_df['stimulus_frames'] = stimulus_frames_
    conditions_df['stimulus_duration_s'] = stimulus_duration_s_
    conditions_df['x_eccentricity_degrees'] = xy_eccentricity_degrees[0]
    conditions_df['y_eccentricity_degrees'] = xy_eccentricity_degrees[1]
    conditions_df['state'] = 'test'
    conditions_df["instruction"] = None
    conditions_df.to_csv('conditions_temp_test.csv', index=False)
    
    # --- Initialize components for Routine "Wait" ---
    # Run 'Begin Experiment' code from PrintTrialNumber
    TRIAL_COUNT = 0 # start counting with 1
    AnyButton = keyboard.Keyboard(deviceName='AnyButton')
    CentralExclusionZone_1 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_1',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-3.0, interpolate=True)
    PressAnyButtonForNextTrialText = visual.TextStim(win=win, name='PressAnyButtonForNextTrialText',
        text='',
        font='Arial',
        units='deg', pos=[0,0], draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "Fixation" ---
    CentralExclusionZone_2 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_2',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-1.0, interpolate=True)
    FixationShape_1 = visual.ShapeStim(
        win=win, name='FixationShape_1',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "Stimulus" ---
    # Run 'Begin Experiment' code from DisplayStimulus
    # Pre-Compute stimuli
    base_xys = {}
    base_oris = {}
    gabor_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute grid
        xs = np.linspace(-spacing * (n_col-1) / 2, spacing * (n_col-1) / 2, n_col)
        ys = np.linspace(0.0, spacing*(n_row_-1), n_row_)
        
        # Pre-compute locations in 3D: (2, n_row, n_col, 2) for [upper/lower, row, col, xy]
        xys_3d = np.zeros((2, n_row_, n_col, 2))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                xys_3d[0, i, j] = [x, y + vertical_seperation / 2]  # upper
                xys_3d[1, i, j] = [x, - y - vertical_seperation / 2]  # lower
        base_xys[n_row_] = xys_3d
        
        # Pre-compute orientations in 3D: (2, n_row, n_col) for [upper/lower, row, col]
        # upper right texture is 0° (vertically), upper left 90° (horizontally) oriented
        # lower right texture is 90° (horizontally), lower left 0° (vertically) oriented
        oris_3d = np.zeros((2, n_row_, n_col)) # all vertical
        oris_3d[0, :, :n_col // 2] = 90  # upper left horizontal
        oris_3d[1, :, n_col // 2:] = 90 # lower right horizontal
        base_oris[n_row_] = oris_3d
        
        # Flatten for ElementArrayStim
        xys_flat = xys_3d.reshape(-1, 2)  # Shape: (2*n_row*n_col, 2)
        oris_flat = oris_3d.flatten()     # Shape: (2*n_row*n_col,)
        
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements=2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            xys=xys_flat,
            sizes=spacing,
            sfs=frequency,
            oris=oris_flat,
            contrs=mitchell_contrast
        )
        gabor_arrays[n_row_] = gabor_array
    # Run 'Begin Experiment' code from DisplayMask
    # Pre-compute scrambled stimulus as masks
    mask_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements= 2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            # maskParams={"sd": 3}, # sets automatically sd = sigma if spacing == 6 sigma 
            xys=base_xys[n_row_].reshape(-1, 2), 
            sizes=spacing,
            sfs=frequency,
            oris=base_oris[n_row_].flatten(), # will be overwriten at begin routine
            contrs=mitchell_contrast
        )
        mask_arrays[n_row_] = gabor_array
    
    CentralExclusionZone_3 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_3',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0, 0, 0],
        opacity=None, depth=-4.0, interpolate=True)
    FixationShape_2 = visual.ShapeStim(
        win=win, name='FixationShape_2',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    GetResponse = keyboard.Keyboard(deviceName='GetResponse')
    # Run 'Begin Experiment' code from SimulateResponse
    PSE_monkey = 0 # columns
    STD_monkey = 0.5 # columns
    RT_monkey = 0.2 # s
    
    # --- Initialize components for Routine "Blank" ---
    
    # --- Initialize components for Routine "Dialogue" ---
    RepeatOrContinueText = visual.TextStim(win=win, name='RepeatOrContinueText',
        text='To repeat, press "r". \n\nTo continue, press "c".',
        font='Arial',
        units='deg', pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    RepeatOrContinueButton = keyboard.Keyboard(deviceName='RepeatOrContinueButton')
    
    # --- Initialize components for Routine "StartTimer" ---
    
    # --- Initialize components for Routine "Wait" ---
    # Run 'Begin Experiment' code from PrintTrialNumber
    TRIAL_COUNT = 0 # start counting with 1
    AnyButton = keyboard.Keyboard(deviceName='AnyButton')
    CentralExclusionZone_1 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_1',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-3.0, interpolate=True)
    PressAnyButtonForNextTrialText = visual.TextStim(win=win, name='PressAnyButtonForNextTrialText',
        text='',
        font='Arial',
        units='deg', pos=[0,0], draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "Fixation" ---
    CentralExclusionZone_2 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_2',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-1.0, interpolate=True)
    FixationShape_1 = visual.ShapeStim(
        win=win, name='FixationShape_1',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "Stimulus" ---
    # Run 'Begin Experiment' code from DisplayStimulus
    # Pre-Compute stimuli
    base_xys = {}
    base_oris = {}
    gabor_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute grid
        xs = np.linspace(-spacing * (n_col-1) / 2, spacing * (n_col-1) / 2, n_col)
        ys = np.linspace(0.0, spacing*(n_row_-1), n_row_)
        
        # Pre-compute locations in 3D: (2, n_row, n_col, 2) for [upper/lower, row, col, xy]
        xys_3d = np.zeros((2, n_row_, n_col, 2))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                xys_3d[0, i, j] = [x, y + vertical_seperation / 2]  # upper
                xys_3d[1, i, j] = [x, - y - vertical_seperation / 2]  # lower
        base_xys[n_row_] = xys_3d
        
        # Pre-compute orientations in 3D: (2, n_row, n_col) for [upper/lower, row, col]
        # upper right texture is 0° (vertically), upper left 90° (horizontally) oriented
        # lower right texture is 90° (horizontally), lower left 0° (vertically) oriented
        oris_3d = np.zeros((2, n_row_, n_col)) # all vertical
        oris_3d[0, :, :n_col // 2] = 90  # upper left horizontal
        oris_3d[1, :, n_col // 2:] = 90 # lower right horizontal
        base_oris[n_row_] = oris_3d
        
        # Flatten for ElementArrayStim
        xys_flat = xys_3d.reshape(-1, 2)  # Shape: (2*n_row*n_col, 2)
        oris_flat = oris_3d.flatten()     # Shape: (2*n_row*n_col,)
        
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements=2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            xys=xys_flat,
            sizes=spacing,
            sfs=frequency,
            oris=oris_flat,
            contrs=mitchell_contrast
        )
        gabor_arrays[n_row_] = gabor_array
    # Run 'Begin Experiment' code from DisplayMask
    # Pre-compute scrambled stimulus as masks
    mask_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements= 2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            # maskParams={"sd": 3}, # sets automatically sd = sigma if spacing == 6 sigma 
            xys=base_xys[n_row_].reshape(-1, 2), 
            sizes=spacing,
            sfs=frequency,
            oris=base_oris[n_row_].flatten(), # will be overwriten at begin routine
            contrs=mitchell_contrast
        )
        mask_arrays[n_row_] = gabor_array
    
    CentralExclusionZone_3 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_3',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0, 0, 0],
        opacity=None, depth=-4.0, interpolate=True)
    FixationShape_2 = visual.ShapeStim(
        win=win, name='FixationShape_2',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    GetResponse = keyboard.Keyboard(deviceName='GetResponse')
    # Run 'Begin Experiment' code from SimulateResponse
    PSE_monkey = 0 # columns
    STD_monkey = 0.5 # columns
    RT_monkey = 0.2 # s
    
    # --- Initialize components for Routine "Blank" ---
    
    # --- Initialize components for Routine "Dialogue" ---
    RepeatOrContinueText = visual.TextStim(win=win, name='RepeatOrContinueText',
        text='To repeat, press "r". \n\nTo continue, press "c".',
        font='Arial',
        units='deg', pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    RepeatOrContinueButton = keyboard.Keyboard(deviceName='RepeatOrContinueButton')
    
    # --- Initialize components for Routine "Wait" ---
    # Run 'Begin Experiment' code from PrintTrialNumber
    TRIAL_COUNT = 0 # start counting with 1
    AnyButton = keyboard.Keyboard(deviceName='AnyButton')
    CentralExclusionZone_1 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_1',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-3.0, interpolate=True)
    PressAnyButtonForNextTrialText = visual.TextStim(win=win, name='PressAnyButtonForNextTrialText',
        text='',
        font='Arial',
        units='deg', pos=[0,0], draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "Fixation" ---
    CentralExclusionZone_2 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_2',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-1.0, interpolate=True)
    FixationShape_1 = visual.ShapeStim(
        win=win, name='FixationShape_1',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "Stimulus" ---
    # Run 'Begin Experiment' code from DisplayStimulus
    # Pre-Compute stimuli
    base_xys = {}
    base_oris = {}
    gabor_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute grid
        xs = np.linspace(-spacing * (n_col-1) / 2, spacing * (n_col-1) / 2, n_col)
        ys = np.linspace(0.0, spacing*(n_row_-1), n_row_)
        
        # Pre-compute locations in 3D: (2, n_row, n_col, 2) for [upper/lower, row, col, xy]
        xys_3d = np.zeros((2, n_row_, n_col, 2))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                xys_3d[0, i, j] = [x, y + vertical_seperation / 2]  # upper
                xys_3d[1, i, j] = [x, - y - vertical_seperation / 2]  # lower
        base_xys[n_row_] = xys_3d
        
        # Pre-compute orientations in 3D: (2, n_row, n_col) for [upper/lower, row, col]
        # upper right texture is 0° (vertically), upper left 90° (horizontally) oriented
        # lower right texture is 90° (horizontally), lower left 0° (vertically) oriented
        oris_3d = np.zeros((2, n_row_, n_col)) # all vertical
        oris_3d[0, :, :n_col // 2] = 90  # upper left horizontal
        oris_3d[1, :, n_col // 2:] = 90 # lower right horizontal
        base_oris[n_row_] = oris_3d
        
        # Flatten for ElementArrayStim
        xys_flat = xys_3d.reshape(-1, 2)  # Shape: (2*n_row*n_col, 2)
        oris_flat = oris_3d.flatten()     # Shape: (2*n_row*n_col,)
        
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements=2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            xys=xys_flat,
            sizes=spacing,
            sfs=frequency,
            oris=oris_flat,
            contrs=mitchell_contrast
        )
        gabor_arrays[n_row_] = gabor_array
    # Run 'Begin Experiment' code from DisplayMask
    # Pre-compute scrambled stimulus as masks
    mask_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements= 2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            # maskParams={"sd": 3}, # sets automatically sd = sigma if spacing == 6 sigma 
            xys=base_xys[n_row_].reshape(-1, 2), 
            sizes=spacing,
            sfs=frequency,
            oris=base_oris[n_row_].flatten(), # will be overwriten at begin routine
            contrs=mitchell_contrast
        )
        mask_arrays[n_row_] = gabor_array
    
    CentralExclusionZone_3 = visual.ShapeStim(
        win=win, name='CentralExclusionZone_3',units='deg', 
        size=(central_exclusion_width_degrees, central_exclusion_height_degrees), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor=[0, 0, 0],
        opacity=None, depth=-4.0, interpolate=True)
    FixationShape_2 = visual.ShapeStim(
        win=win, name='FixationShape_2',units='deg', 
        size=(fixation_diameter, fixation_diameter), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    GetResponse = keyboard.Keyboard(deviceName='GetResponse')
    # Run 'Begin Experiment' code from SimulateResponse
    PSE_monkey = 0 # columns
    STD_monkey = 0.5 # columns
    RT_monkey = 0.2 # s
    
    # --- Initialize components for Routine "Blank" ---
    
    # --- Initialize components for Routine "Break" ---
    TakeABreakText = visual.TextStim(win=win, name='TakeABreakText',
        text='Take a break!',
        font='Arial',
        units='deg', pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    AnyButton_2 = keyboard.Keyboard(deviceName='AnyButton_2')
    
    # --- Initialize components for Routine "QandASession" ---
    win.allowStencil = True
    QandAForm = visual.Form(win=win, name='QandAForm',
        items='C:/Users/kai.rothe/Documents/V1SH_experiment/experiment/QandA.csv',
        textHeight=0.03,
        font='Noto Sans',
        randomize=False,
        style='dark',
        fillColor=None, borderColor=None, itemColor='white', 
        responseColor='white', markerColor='red', colorSpace='rgb', 
        size=(1.0, 0.9),
        pos=(-0.3, 0),
        itemPadding=0.05,
        depth=-1
    )
    FinishButton = visual.ButtonStim(win, 
        text='Click HERE to finish', font='Arial',
        pos=(10, 0),units='deg',
        letterHeight=1.0,
        size=(7, 8), 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='red', borderColor='red',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=False, italic=False,
        padding=None,
        anchor='center',
        name='FinishButton',
        depth=-2
    )
    FinishButton.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "SaveData" ---
    # Run 'Begin Experiment' code from SaveAdditionalData
    def save_experiment_files_robust():
        """Save experiment files using PsychoPy's auto-generated filename"""
        try:
            # 'filename' is auto-generated by Builder from Experiment Settings
            # It's already the full path without extension
            # E.g., 'data/participant001_myExperiment_2025-11-20_1435'
            dataDir = os.path.dirname(filename)
            baseName = os.path.basename(filename) 
            
            # Files to save: (relative_path, original_filename, extension)
            files_to_save = [
                ("", expName + '.psyexp', '.psyexp'),
                ("..", "instructions.csv", '.csv'),
                ("..", "ExpInfo.yaml", '.yaml'),
                ("..", "QandA.csv", ".csv")
            ]
            
            for relative_path, source_filename, ext in files_to_save:
                # Build the full source path correctly
                if relative_path:
                    filepath = os.path.join(_thisDir, relative_path, source_filename)
                else:
                    filepath = os.path.join(_thisDir, source_filename)
                
                # Normalize the path to handle ".." properly
                filepath = os.path.normpath(filepath)
                
                print(f"Looking for: {filepath}")  # Debug print
                
                if os.path.exists(filepath):
                    # Get modification date
                    modDate = datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).strftime('%Y-%m-%d_%H%M')
                    
                    # Extract just the name without extension for renaming
                    source_base = os.path.splitext(source_filename)[0]
                    
                    # Create destination filename
                    destFilename = f"{source_base}_mod-{modDate}{ext}"
                    destPath = os.path.join(dataDir, destFilename)
                    
                    shutil.copy2(filepath, destPath)
                    print(f"Saved: {destFilename}")
                else:
                    print(f"File not found: {filepath}")  # Debug print
                    
        except Exception as e:
            logging.error(f'Could not save experiment files: {e}')
            print(f"Error in save_experiment_files_robust: {e}")  # Additional debug
    
    def get_psychopy_info():
        # Function to dynamically retrieve PsychoPy information
        psychopy_info = {
            "Hardware": {
                "Display": {
                    "Type": win.monitor.name,
                    "MeanLuminance_cd/m2": expInfo['mean_luminance_cdpm2'],
                    "RefreshRate_Hz": expInfo['frameRate'],
                    "ViewingDistance_cm": expInfo["viewing_distance_cm"],
                    "GammaCorrection": win.monitor.getGammaGrid() is not None
                },
                "EyeTracker": {
                    "Usage": False # TODO: Replace with actual eye tracker check
                }
            }
        }
        return psychopy_info
    
    def compare_and_update(yaml_file, psychopy_info):
        # Compare psychopy hardware with manual ExpInfo YAML file
        # and update values if necessary
        updated = False
        for section, values in psychopy_info.items():
            if section in yaml_file:
                for key, value in values.items():
                    if key in yaml_file[section] and yaml_file[section][key] != value:
                        yaml_file[section][key] = value
                        if key != "RandomSeed":
                            updated = True
                        
            else:
                yaml_file[section] = values
                updated = True
        
        if updated:  
            print("Warning: ExpInfo file modified.")
            logging.warning("ExpInfo file modified.")
    
        return updated
    
    # Save staircase updates
    def eval_and_save_staircase():
        if not staircases: 
            print("Warning: eval_and_save_staircase could not be executed (no staircases initialized)")
            logging.warning("eval_and_save_staircase could not be executed (no staircases initialized)")
            return 
        else: 
            try:
                for label, staircase_ in staircases.items(): # name "staircase_" to avoid overwriting global STAIRCASE variable
                    # 1. Print out and save parameters
                    params_dict =  staircase_.param_estimate
                    mean, std, lapse_rate = params_dict["mean"], params_dict["sd"], params_dict["lapse_rate"]
                    print(f"Parameter of staircase {label}:\n   mean = {mean}\n   std = {std}\n   lapse_rate = {lapse_rate}")
                    
                    # 'filename' is auto-generated by Builder from Experiment Settings
                    # It's already the full path without extension
                    # E.g., 'data/participant001_myExperiment_2025-11-20_1435'
                    dataDir = os.path.dirname(filename)
                    results_name = "staircase_params" + "_" + label + ".json"
                    save_as = os.path.join(dataDir, results_name)
                    with open(save_as, 'w') as f:
                        json.dump(params_dict, f, indent=2)
                    print(f"Saved: {results_name}")
                    
                    # 2. Save complete staircase for reloading e.g. detailed analysis
                    staircase_name = "questplus_staircase" + "_" + label + ".json"
                    save_as = os.path.join(dataDir, staircase_name)
                    staircase_json_dump = staircase_.to_json() # string
                    with open(save_as, 'w') as f:
                        f.write(staircase_json_dump)
                    print(f"Saved: {staircase_name}")
                    
                    # 3. Save marginal posteriors
                    posterior_name = "questplus_posterior" + "_" + label + ".pkl"
                    save_as = os.path.join(dataDir, posterior_name)
                    posterior_dict = staircase_.marginal_posterior
                    with open(save_as, 'wb') as f:
                        pickle.dump(posterior_dict, f)
                    print(f"Saved: {posterior_name}")
                
            except Exception as e:
                raise(e)
                # logging.error(f'Could not save quest eval: {e}')
                # print(f"Error in eval_and_save_staircase: {e}")  # Additional debug
    
    # Update ExpInfo with PsychoPy hardware info
    psychopy_info = get_psychopy_info()
    compare_and_update(ExpInfo, psychopy_info)
    
    # Save immediately at start
    save_experiment_files_robust()
    
    # Save at the end, or if computer fails / experiment is stopped by "esc"
    atexit.register(save_experiment_files_robust)
    atexit.register(eval_and_save_staircase)
    
    # --- Initialize components for Routine "End" ---
    ThankYou = visual.TextStim(win=win, name='ThankYou',
        text='Thank you!',
        font='Arial',
        units='deg', pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Preparation" ---
    # create an object to store info about Routine Preparation
    Preparation = data.Routine(
        name='Preparation',
        components=[],
    )
    Preparation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from GetExperimentalInfo
    trial_variables = {"n_row": None,
        "displacement_cols": None,
        "orientation_degrees": None,
        "stimulus_duration_s": None,
        "instruction": None,
        "session": None,
        "stimulus_frames": None,
        "x_eccentricity_degrees": None,
        "y_eccentricity_degrees": None,
        "state": None,
        "response": None}
    # store start times for Preparation
    Preparation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Preparation.tStart = globalClock.getTime(format='float')
    Preparation.status = STARTED
    thisExp.addData('Preparation.started', Preparation.tStart)
    Preparation.maxDuration = None
    # keep track of which components have finished
    PreparationComponents = Preparation.components
    for thisComponent in Preparation.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Preparation" ---
    Preparation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Preparation,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Preparation.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Preparation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Preparation" ---
    for thisComponent in Preparation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Preparation
    Preparation.tStop = globalClock.getTime(format='float')
    Preparation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Preparation.stopped', Preparation.tStop)
    thisExp.nextEntry()
    # the Routine "Preparation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    instruction = data.TrialHandler2(
        name='instruction',
        nReps=999 if expInfo['skip_instructions'] == False else 0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(instruction)  # add the loop to the experiment
    thisInstruction = instruction.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruction.rgb)
    if thisInstruction != None:
        for paramName in thisInstruction:
            globals()[paramName] = thisInstruction[paramName]
    
    for thisInstruction in instruction:
        instruction.status = STARTED
        if hasattr(thisInstruction, 'status'):
            thisInstruction.status = STARTED
        currentLoop = instruction
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisInstruction.rgb)
        if thisInstruction != None:
            for paramName in thisInstruction:
                globals()[paramName] = thisInstruction[paramName]
        
        # set up handler to look after randomisation of conditions etc
        instruction_trials = data.TrialHandler2(
            name='instruction_trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditions_temp_instruct.csv'), 
            seed=None, 
        )
        thisExp.addLoop(instruction_trials)  # add the loop to the experiment
        thisInstruction_trial = instruction_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInstruction_trial.rgb)
        if thisInstruction_trial != None:
            for paramName in thisInstruction_trial:
                globals()[paramName] = thisInstruction_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisInstruction_trial in instruction_trials:
            instruction_trials.status = STARTED
            if hasattr(thisInstruction_trial, 'status'):
                thisInstruction_trial.status = STARTED
            currentLoop = instruction_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisInstruction_trial.rgb)
            if thisInstruction_trial != None:
                for paramName in thisInstruction_trial:
                    globals()[paramName] = thisInstruction_trial[paramName]
            
            # --- Prepare to start Routine "Wait" ---
            # create an object to store info about Routine Wait
            Wait = data.Routine(
                name='Wait',
                components=[AnyButton, CentralExclusionZone_1, PressAnyButtonForNextTrialText],
            )
            Wait.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground
            win.color = [0, 0, 0] # assure grey background
            # Run 'Begin Routine' code from PrintTrialNumber
            TRIAL_COUNT += 1
            print(f"{state} trial {TRIAL_COUNT}") # print to note down and refind errors
            thisExp.addData('global.thisN', TRIAL_COUNT) # save to compare with noted down global trial number
            # create starting attributes for AnyButton
            AnyButton.keys = []
            AnyButton.rt = []
            _AnyButton_allKeys = []
            CentralExclusionZone_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            PressAnyButtonForNextTrialText.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            PressAnyButtonForNextTrialText.setText("Press any button\nfor %d. trial" % (TRIAL_COUNT))
            # store start times for Wait
            Wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Wait.tStart = globalClock.getTime(format='float')
            Wait.status = STARTED
            thisExp.addData('Wait.started', Wait.tStart)
            Wait.maxDuration = None
            # keep track of which components have finished
            WaitComponents = Wait.components
            for thisComponent in Wait.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Wait" ---
            Wait.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstruction_trial, 'status') and thisInstruction_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *AnyButton* updates
                waitOnFlip = False
                
                # if AnyButton is starting this frame...
                if AnyButton.status == NOT_STARTED and frameN >= 10:
                    # keep track of start time/frame for later
                    AnyButton.frameNStart = frameN  # exact frame index
                    AnyButton.tStart = t  # local t and not account for scr refresh
                    AnyButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(AnyButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AnyButton.started')
                    # update status
                    AnyButton.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(AnyButton.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(AnyButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if AnyButton.status == STARTED and not waitOnFlip:
                    theseKeys = AnyButton.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                    _AnyButton_allKeys.extend(theseKeys)
                    if len(_AnyButton_allKeys):
                        AnyButton.keys = _AnyButton_allKeys[-1].name  # just the last key pressed
                        AnyButton.rt = _AnyButton_allKeys[-1].rt
                        AnyButton.duration = _AnyButton_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *CentralExclusionZone_1* updates
                
                # if CentralExclusionZone_1 is starting this frame...
                if CentralExclusionZone_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_1.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_1.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_1.started')
                    # update status
                    CentralExclusionZone_1.status = STARTED
                    CentralExclusionZone_1.setAutoDraw(True)
                
                # if CentralExclusionZone_1 is active this frame...
                if CentralExclusionZone_1.status == STARTED:
                    # update params
                    pass
                
                # *PressAnyButtonForNextTrialText* updates
                
                # if PressAnyButtonForNextTrialText is starting this frame...
                if PressAnyButtonForNextTrialText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    PressAnyButtonForNextTrialText.frameNStart = frameN  # exact frame index
                    PressAnyButtonForNextTrialText.tStart = t  # local t and not account for scr refresh
                    PressAnyButtonForNextTrialText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(PressAnyButtonForNextTrialText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PressAnyButtonForNextTrialText.started')
                    # update status
                    PressAnyButtonForNextTrialText.status = STARTED
                    PressAnyButtonForNextTrialText.setAutoDraw(True)
                
                # if PressAnyButtonForNextTrialText is active this frame...
                if PressAnyButtonForNextTrialText.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from SimulateButtonPress
                if expInfo.get('monkey_mode', False):
                    # simulate response if no key was pressed yet
                    if not AnyButton.keys:
                        if t > RT_monkey:
                            AnyButton.keys = "enter"
                            AnyButton.rt = RT_monkey
                            AnyButton.duration = None
                            continueRoutine = False 
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Wait,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Wait.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Wait.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Wait" ---
            for thisComponent in Wait.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Wait
            Wait.tStop = globalClock.getTime(format='float')
            Wait.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Wait.stopped', Wait.tStop)
            # check responses
            if AnyButton.keys in ['', [], None]:  # No response was made
                AnyButton.keys = None
            instruction_trials.addData('AnyButton.keys',AnyButton.keys)
            if AnyButton.keys != None:  # we had a response
                instruction_trials.addData('AnyButton.rt', AnyButton.rt)
                instruction_trials.addData('AnyButton.duration', AnyButton.duration)
            # the Routine "Wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Fixation" ---
            # create an object to store info about Routine Fixation
            Fixation = data.Routine(
                name='Fixation',
                components=[CentralExclusionZone_2, FixationShape_1],
            )
            Fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_2
            win.color = [0, 0, 0] # keep grey background
            CentralExclusionZone_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            FixationShape_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            # store start times for Fixation
            Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Fixation.tStart = globalClock.getTime(format='float')
            Fixation.status = STARTED
            thisExp.addData('Fixation.started', Fixation.tStart)
            # keep track of which components have finished
            FixationComponents = Fixation.components
            for thisComponent in Fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Fixation" ---
            Fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstruction_trial, 'status') and thisInstruction_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= fixation_frames:
                    continueRoutine = False
                
                # *CentralExclusionZone_2* updates
                
                # if CentralExclusionZone_2 is starting this frame...
                if CentralExclusionZone_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_2.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_2.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_2.started')
                    # update status
                    CentralExclusionZone_2.status = STARTED
                    CentralExclusionZone_2.setAutoDraw(True)
                
                # if CentralExclusionZone_2 is active this frame...
                if CentralExclusionZone_2.status == STARTED:
                    # update params
                    pass
                
                # *FixationShape_1* updates
                
                # if FixationShape_1 is starting this frame...
                if FixationShape_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationShape_1.frameNStart = frameN  # exact frame index
                    FixationShape_1.tStart = t  # local t and not account for scr refresh
                    FixationShape_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationShape_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationShape_1.started')
                    # update status
                    FixationShape_1.status = STARTED
                    FixationShape_1.setAutoDraw(True)
                
                # if FixationShape_1 is active this frame...
                if FixationShape_1.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Fixation,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Fixation" ---
            for thisComponent in Fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Fixation
            Fixation.tStop = globalClock.getTime(format='float')
            Fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Fixation.stopped', Fixation.tStop)
            # the Routine "Fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Stimulus" ---
            # create an object to store info about Routine Stimulus
            Stimulus = data.Routine(
                name='Stimulus',
                components=[CentralExclusionZone_3, FixationShape_2, GetResponse],
            )
            Stimulus.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_3
            win.color = [0, 0, 0] # keep grey background
            # Run 'Begin Routine' code from UpdateVariablesAndStaircase
            # Update trials variables (from globals)
            for variable_name in trial_variables.keys():
                if variable_name != "response":
                    trial_variables[variable_name] = globals().get(variable_name, None)
            
            # Infer next horizontal displacement
            if trial_variables["state"] == "test": # staircase during testing trials
                key = get_key(trial_variables["n_row"], trial_variables["orientation_degrees"])
                STAIRCASE = staircases[key] # overwrite from previous trial (and "none" initially)
                STIMULUS_DICT = STAIRCASE.next_stim
                trial_variables["displacement_cols"] = STIMULUS_DICT["intensity"]
                if trial_variables["orientation_degrees"] >= 90: # one staircase for symmetric oriented textures
                    trial_variables["displacement_cols"] = - trial_variables["displacement_cols"]
            # Run 'Begin Routine' code from DisplayStimulus
            win.color = [0, 0, 0] # keep grey background
            
            # randomize location of Gabor patch centers
            jitter = rng.uniform(-jitter_amount, jitter_amount, 
                                       size=(2, trial_variables["n_row"], n_col, 2))
            xys_3D = base_xys[trial_variables["n_row"]] + jitter
            
            # rotate to set orientation
            oris_3D = base_oris[trial_variables["n_row"]] + trial_variables["orientation_degrees"]
            
            # displace upper texture horizontally
            xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
            
            # set eccentricity of texture
            xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"] 
            xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"]
            
            # set and update stimulus
            gabor_array = gabor_arrays[trial_variables["n_row"]]
            gabor_array.xys = xys_3D.reshape(-1, 2)
            gabor_array.oris = oris_3D.flatten()
            # Run 'Begin Routine' code from DisplayMask
            # Set mask
            mask_array = mask_arrays[trial_variables["n_row"]]
            
            # Copy stimulus locations to avoid confusing salient shifts after stimulus
            xys_3D = base_xys[trial_variables["n_row"]] + jitter
            xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
            xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"]
            xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"] 
            mask_array.xys = xys_3D.reshape(-1, 2)
            
            # But scramble orientation and thus texture
            mask_array.oris = rng.uniform(0, 180, size=2 * trial_variables["n_row"] * n_col)
            CentralExclusionZone_3.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            FixationShape_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            # create starting attributes for GetResponse
            GetResponse.keys = []
            GetResponse.rt = []
            _GetResponse_allKeys = []
            # store start times for Stimulus
            Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Stimulus.tStart = globalClock.getTime(format='float')
            Stimulus.status = STARTED
            thisExp.addData('Stimulus.started', Stimulus.tStart)
            Stimulus.maxDuration = None
            # keep track of which components have finished
            StimulusComponents = Stimulus.components
            for thisComponent in Stimulus.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Stimulus" ---
            Stimulus.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstruction_trial, 'status') and thisInstruction_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from DisplayStimulus
                if frameN < trial_variables["stimulus_frames"]:
                    gabor_array.draw()
                # Run 'Each Frame' code from DisplayMask
                if frameN >= trial_variables["stimulus_frames"]:
                    mask_array.draw()
                
                # *CentralExclusionZone_3* updates
                
                # if CentralExclusionZone_3 is starting this frame...
                if CentralExclusionZone_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_3.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_3.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_3.started')
                    # update status
                    CentralExclusionZone_3.status = STARTED
                    CentralExclusionZone_3.setAutoDraw(True)
                
                # if CentralExclusionZone_3 is active this frame...
                if CentralExclusionZone_3.status == STARTED:
                    # update params
                    pass
                
                # *FixationShape_2* updates
                
                # if FixationShape_2 is starting this frame...
                if FixationShape_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationShape_2.frameNStart = frameN  # exact frame index
                    FixationShape_2.tStart = t  # local t and not account for scr refresh
                    FixationShape_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationShape_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationShape_2.started')
                    # update status
                    FixationShape_2.status = STARTED
                    FixationShape_2.setAutoDraw(True)
                
                # if FixationShape_2 is active this frame...
                if FixationShape_2.status == STARTED:
                    # update params
                    pass
                
                # *GetResponse* updates
                waitOnFlip = False
                
                # if GetResponse is starting this frame...
                if GetResponse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    GetResponse.frameNStart = frameN  # exact frame index
                    GetResponse.tStart = t  # local t and not account for scr refresh
                    GetResponse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(GetResponse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'GetResponse.started')
                    # update status
                    GetResponse.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(GetResponse.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(GetResponse.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if GetResponse.status == STARTED and not waitOnFlip:
                    theseKeys = GetResponse.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _GetResponse_allKeys.extend(theseKeys)
                    if len(_GetResponse_allKeys):
                        GetResponse.keys = _GetResponse_allKeys[-1].name  # just the last key pressed
                        GetResponse.rt = _GetResponse_allKeys[-1].rt
                        GetResponse.duration = _GetResponse_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from SimulateResponse
                if expInfo.get('monkey_mode', False):
                    # simulate response if no key was pressed yet
                    if not GetResponse.keys:
                        # wait realistrically
                        if t > RT_monkey:  
                            simulated_key = simulate_response(
                                trial_variables["displacement_cols"], 
                                mu=PSE_monkey, 
                                sigma=STD_monkey
                            )
                            
                            GetResponse.keys = simulated_key
                            GetResponse.rt = RT_monkey
                            GetResponse.duration = None
                            continueRoutine = False  # force routine end
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Stimulus,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Stimulus.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Stimulus.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Stimulus" ---
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Stimulus
            Stimulus.tStop = globalClock.getTime(format='float')
            Stimulus.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Stimulus.stopped', Stimulus.tStop)
            # Run 'End Routine' code from UpdateVariablesAndStaircase
            # Update staircase
            if trial_variables["state"] == "test":
                # staircase: positive == towards the upper texture parallely oriented to border
                if trial_variables["orientation_degrees"] < 90: # right upper texture more parallel to border
                    response = 1 if (GetResponse.keys == "right") else 0
                elif trial_variables["orientation_degrees"] >= 90: # left upper texture more parallel to border
                    response = 1 if (GetResponse.keys == "left") else 0
                STAIRCASE.update(stim=STIMULUS_DICT, outcome={"response": response}) # variable "staircase" and "stimulus_dict" already selected at begin routine
            # Run 'End Routine' code from DisplayMask
            win.color = [0, 0, 0] # keep grey background
            # check responses
            if GetResponse.keys in ['', [], None]:  # No response was made
                GetResponse.keys = None
            instruction_trials.addData('GetResponse.keys',GetResponse.keys)
            if GetResponse.keys != None:  # we had a response
                instruction_trials.addData('GetResponse.rt', GetResponse.rt)
                instruction_trials.addData('GetResponse.duration', GetResponse.duration)
            # Run 'End Routine' code from SaveTrialData
            # save trial to trial data 
            # possibly redundant for non-staircase loops
            trial_variables["response"] = GetResponse.keys
            for variable_name, variable_value in trial_variables.items():
                thisExp.addData(variable_name, variable_value) 
            # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Blank" ---
            # create an object to store info about Routine Blank
            Blank = data.Routine(
                name='Blank',
                components=[],
            )
            Blank.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_4
            win.clearBuffer() 
            win.color = [0, 0, 0] # keep grey background
            # store start times for Blank
            Blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Blank.tStart = globalClock.getTime(format='float')
            Blank.status = STARTED
            thisExp.addData('Blank.started', Blank.tStart)
            # keep track of which components have finished
            BlankComponents = Blank.components
            for thisComponent in Blank.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Blank" ---
            Blank.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstruction_trial, 'status') and thisInstruction_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= blank_frames:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Blank,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Blank.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Blank.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Blank" ---
            for thisComponent in Blank.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Blank
            Blank.tStop = globalClock.getTime(format='float')
            Blank.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Blank.stopped', Blank.tStop)
            # Run 'End Routine' code from CheckTimeAndTrialNumber
            # check clock
            if track_time and trial_variables["state"] == "test":
                elapsed_time_s = time.time() - practice_start_time_s
                if elapsed_time_s >= MAX_TIME_s:
                    # log
                    message = f"Time limit reached: {elapsed_time_s/60:.1f} minutes elapsed after {testing_trials.thisN + 1} testing trials"
                    print(message)
                    logging.info(message)
                    logging.exp(message)
                    
                    # Add summary data
                    thisExp.addData('time_limit_min', MAX_TIME_s / 60)
                    thisExp.addData('tracked_time_min', elapsed_time_s / 60)
                    thisExp.addData('time_limit_was_reached', True)
                    
                    # finish experiment
                    testing_trials.finished = True
            
            # check trial number
            if trial_variables["state"] == "train": # for some valid ExpInfo.yaml settings, there are more training conditions then nTrials_train
                if training_trials.thisN + 1 >= nTrials_train: # + 1 since thisN starts at 0
                    training_trials.finished = True
            elif not track_time and trial_variables["state"] == "test": # if not track time, assure right number of trials despite of rounding issues
                if testing_trials.thisN + 1 >= nTrials_test: # + 1 since thisN starts at 0
                    testing_trials.finished = True
            # the Routine "Blank" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisInstruction_trial as finished
            if hasattr(thisInstruction_trial, 'status'):
                thisInstruction_trial.status = FINISHED
            # if awaiting a pause, pause now
            if instruction_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                instruction_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'instruction_trials'
        instruction_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if instruction_trials.trialList in ([], [None], None):
            params = []
        else:
            params = instruction_trials.trialList[0].keys()
        # save data for this loop
        instruction_trials.saveAsExcel(filename + '.xlsx', sheetName='instruction_trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        instruction_trials.saveAsText(filename + '_instruction_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "Dialogue" ---
        # create an object to store info about Routine Dialogue
        Dialogue = data.Routine(
            name='Dialogue',
            components=[RepeatOrContinueText, RepeatOrContinueButton],
        )
        Dialogue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for RepeatOrContinueButton
        RepeatOrContinueButton.keys = []
        RepeatOrContinueButton.rt = []
        _RepeatOrContinueButton_allKeys = []
        # store start times for Dialogue
        Dialogue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Dialogue.tStart = globalClock.getTime(format='float')
        Dialogue.status = STARTED
        thisExp.addData('Dialogue.started', Dialogue.tStart)
        Dialogue.maxDuration = None
        # keep track of which components have finished
        DialogueComponents = Dialogue.components
        for thisComponent in Dialogue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Dialogue" ---
        Dialogue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisInstruction, 'status') and thisInstruction.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *RepeatOrContinueText* updates
            
            # if RepeatOrContinueText is starting this frame...
            if RepeatOrContinueText.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                RepeatOrContinueText.frameNStart = frameN  # exact frame index
                RepeatOrContinueText.tStart = t  # local t and not account for scr refresh
                RepeatOrContinueText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RepeatOrContinueText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RepeatOrContinueText.started')
                # update status
                RepeatOrContinueText.status = STARTED
                RepeatOrContinueText.setAutoDraw(True)
            
            # if RepeatOrContinueText is active this frame...
            if RepeatOrContinueText.status == STARTED:
                # update params
                pass
            
            # *RepeatOrContinueButton* updates
            waitOnFlip = False
            
            # if RepeatOrContinueButton is starting this frame...
            if RepeatOrContinueButton.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                RepeatOrContinueButton.frameNStart = frameN  # exact frame index
                RepeatOrContinueButton.tStart = t  # local t and not account for scr refresh
                RepeatOrContinueButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RepeatOrContinueButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RepeatOrContinueButton.started')
                # update status
                RepeatOrContinueButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(RepeatOrContinueButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(RepeatOrContinueButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if RepeatOrContinueButton.status == STARTED and not waitOnFlip:
                theseKeys = RepeatOrContinueButton.getKeys(keyList=['r','c'], ignoreKeys=["escape"], waitRelease=False)
                _RepeatOrContinueButton_allKeys.extend(theseKeys)
                if len(_RepeatOrContinueButton_allKeys):
                    RepeatOrContinueButton.keys = _RepeatOrContinueButton_allKeys[-1].name  # just the last key pressed
                    RepeatOrContinueButton.rt = _RepeatOrContinueButton_allKeys[-1].rt
                    RepeatOrContinueButton.duration = _RepeatOrContinueButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Dialogue,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Dialogue.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Dialogue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Dialogue" ---
        for thisComponent in Dialogue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Dialogue
        Dialogue.tStop = globalClock.getTime(format='float')
        Dialogue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Dialogue.stopped', Dialogue.tStop)
        # check responses
        if RepeatOrContinueButton.keys in ['', [], None]:  # No response was made
            RepeatOrContinueButton.keys = None
        instruction.addData('RepeatOrContinueButton.keys',RepeatOrContinueButton.keys)
        if RepeatOrContinueButton.keys != None:  # we had a response
            instruction.addData('RepeatOrContinueButton.rt', RepeatOrContinueButton.rt)
            instruction.addData('RepeatOrContinueButton.duration', RepeatOrContinueButton.duration)
        # Run 'End Routine' code from FinishLoop
        if RepeatOrContinueButton.keys == 'c':
            if trial_variables["state"] == 'instruct': # stop instructions
                instruction.finished = True
            elif trial_variables["state"] == 'train': # stop training / practice
                training.finished = True
        # the Routine "Dialogue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisInstruction as finished
        if hasattr(thisInstruction, 'status'):
            thisInstruction.status = FINISHED
        # if awaiting a pause, pause now
        if instruction.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            instruction.status = STARTED
    # completed 999 if expInfo['skip_instructions'] == False else 0 repeats of 'instruction'
    instruction.status = FINISHED
    
    
    # --- Prepare to start Routine "StartTimer" ---
    # create an object to store info about Routine StartTimer
    StartTimer = data.Routine(
        name='StartTimer',
        components=[],
    )
    StartTimer.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from Timer
    # Start tracking time
    if track_time:
        practice_start_time_s = time.time()
        
        # log
        message = f"Timer started at {practice_start_time_s / 60} minutes"
        print(message)
        logging.info(message)
        logging.exp(message)
        
        # Save summary data
        thisExp.addData("practice_start_time_min", practice_start_time_s / 60)
        thisExp.addData("time_was_tracked", True)
        thisExp.addData('time_limit_was_reached', False)
    # store start times for StartTimer
    StartTimer.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    StartTimer.tStart = globalClock.getTime(format='float')
    StartTimer.status = STARTED
    thisExp.addData('StartTimer.started', StartTimer.tStart)
    StartTimer.maxDuration = None
    # keep track of which components have finished
    StartTimerComponents = StartTimer.components
    for thisComponent in StartTimer.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "StartTimer" ---
    StartTimer.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=StartTimer,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            StartTimer.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartTimer.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "StartTimer" ---
    for thisComponent in StartTimer.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for StartTimer
    StartTimer.tStop = globalClock.getTime(format='float')
    StartTimer.tStopRefresh = tThisFlipGlobal
    thisExp.addData('StartTimer.stopped', StartTimer.tStop)
    thisExp.nextEntry()
    # the Routine "StartTimer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    training = data.TrialHandler2(
        name='training',
        nReps=999 if expInfo['skip_training'] == False else 0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(training)  # add the loop to the experiment
    thisTraining = training.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
    if thisTraining != None:
        for paramName in thisTraining:
            globals()[paramName] = thisTraining[paramName]
    
    for thisTraining in training:
        training.status = STARTED
        if hasattr(thisTraining, 'status'):
            thisTraining.status = STARTED
        currentLoop = training
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
        if thisTraining != None:
            for paramName in thisTraining:
                globals()[paramName] = thisTraining[paramName]
        
        # set up handler to look after randomisation of conditions etc
        training_trials = data.TrialHandler2(
            name='training_trials',
            nReps=nReps_train, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditions_temp_train.csv'), 
            seed=seed + 1, 
        )
        thisExp.addLoop(training_trials)  # add the loop to the experiment
        thisTraining_trial = training_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_trial.rgb)
        if thisTraining_trial != None:
            for paramName in thisTraining_trial:
                globals()[paramName] = thisTraining_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTraining_trial in training_trials:
            training_trials.status = STARTED
            if hasattr(thisTraining_trial, 'status'):
                thisTraining_trial.status = STARTED
            currentLoop = training_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTraining_trial.rgb)
            if thisTraining_trial != None:
                for paramName in thisTraining_trial:
                    globals()[paramName] = thisTraining_trial[paramName]
            
            # --- Prepare to start Routine "Wait" ---
            # create an object to store info about Routine Wait
            Wait = data.Routine(
                name='Wait',
                components=[AnyButton, CentralExclusionZone_1, PressAnyButtonForNextTrialText],
            )
            Wait.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground
            win.color = [0, 0, 0] # assure grey background
            # Run 'Begin Routine' code from PrintTrialNumber
            TRIAL_COUNT += 1
            print(f"{state} trial {TRIAL_COUNT}") # print to note down and refind errors
            thisExp.addData('global.thisN', TRIAL_COUNT) # save to compare with noted down global trial number
            # create starting attributes for AnyButton
            AnyButton.keys = []
            AnyButton.rt = []
            _AnyButton_allKeys = []
            CentralExclusionZone_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            PressAnyButtonForNextTrialText.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            PressAnyButtonForNextTrialText.setText("Press any button\nfor %d. trial" % (TRIAL_COUNT))
            # store start times for Wait
            Wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Wait.tStart = globalClock.getTime(format='float')
            Wait.status = STARTED
            thisExp.addData('Wait.started', Wait.tStart)
            Wait.maxDuration = None
            # keep track of which components have finished
            WaitComponents = Wait.components
            for thisComponent in Wait.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Wait" ---
            Wait.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTraining_trial, 'status') and thisTraining_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *AnyButton* updates
                waitOnFlip = False
                
                # if AnyButton is starting this frame...
                if AnyButton.status == NOT_STARTED and frameN >= 10:
                    # keep track of start time/frame for later
                    AnyButton.frameNStart = frameN  # exact frame index
                    AnyButton.tStart = t  # local t and not account for scr refresh
                    AnyButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(AnyButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'AnyButton.started')
                    # update status
                    AnyButton.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(AnyButton.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(AnyButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if AnyButton.status == STARTED and not waitOnFlip:
                    theseKeys = AnyButton.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                    _AnyButton_allKeys.extend(theseKeys)
                    if len(_AnyButton_allKeys):
                        AnyButton.keys = _AnyButton_allKeys[-1].name  # just the last key pressed
                        AnyButton.rt = _AnyButton_allKeys[-1].rt
                        AnyButton.duration = _AnyButton_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *CentralExclusionZone_1* updates
                
                # if CentralExclusionZone_1 is starting this frame...
                if CentralExclusionZone_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_1.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_1.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_1.started')
                    # update status
                    CentralExclusionZone_1.status = STARTED
                    CentralExclusionZone_1.setAutoDraw(True)
                
                # if CentralExclusionZone_1 is active this frame...
                if CentralExclusionZone_1.status == STARTED:
                    # update params
                    pass
                
                # *PressAnyButtonForNextTrialText* updates
                
                # if PressAnyButtonForNextTrialText is starting this frame...
                if PressAnyButtonForNextTrialText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    PressAnyButtonForNextTrialText.frameNStart = frameN  # exact frame index
                    PressAnyButtonForNextTrialText.tStart = t  # local t and not account for scr refresh
                    PressAnyButtonForNextTrialText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(PressAnyButtonForNextTrialText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PressAnyButtonForNextTrialText.started')
                    # update status
                    PressAnyButtonForNextTrialText.status = STARTED
                    PressAnyButtonForNextTrialText.setAutoDraw(True)
                
                # if PressAnyButtonForNextTrialText is active this frame...
                if PressAnyButtonForNextTrialText.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from SimulateButtonPress
                if expInfo.get('monkey_mode', False):
                    # simulate response if no key was pressed yet
                    if not AnyButton.keys:
                        if t > RT_monkey:
                            AnyButton.keys = "enter"
                            AnyButton.rt = RT_monkey
                            AnyButton.duration = None
                            continueRoutine = False 
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Wait,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Wait.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Wait.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Wait" ---
            for thisComponent in Wait.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Wait
            Wait.tStop = globalClock.getTime(format='float')
            Wait.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Wait.stopped', Wait.tStop)
            # check responses
            if AnyButton.keys in ['', [], None]:  # No response was made
                AnyButton.keys = None
            training_trials.addData('AnyButton.keys',AnyButton.keys)
            if AnyButton.keys != None:  # we had a response
                training_trials.addData('AnyButton.rt', AnyButton.rt)
                training_trials.addData('AnyButton.duration', AnyButton.duration)
            # the Routine "Wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Fixation" ---
            # create an object to store info about Routine Fixation
            Fixation = data.Routine(
                name='Fixation',
                components=[CentralExclusionZone_2, FixationShape_1],
            )
            Fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_2
            win.color = [0, 0, 0] # keep grey background
            CentralExclusionZone_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            FixationShape_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            # store start times for Fixation
            Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Fixation.tStart = globalClock.getTime(format='float')
            Fixation.status = STARTED
            thisExp.addData('Fixation.started', Fixation.tStart)
            # keep track of which components have finished
            FixationComponents = Fixation.components
            for thisComponent in Fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Fixation" ---
            Fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTraining_trial, 'status') and thisTraining_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= fixation_frames:
                    continueRoutine = False
                
                # *CentralExclusionZone_2* updates
                
                # if CentralExclusionZone_2 is starting this frame...
                if CentralExclusionZone_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_2.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_2.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_2.started')
                    # update status
                    CentralExclusionZone_2.status = STARTED
                    CentralExclusionZone_2.setAutoDraw(True)
                
                # if CentralExclusionZone_2 is active this frame...
                if CentralExclusionZone_2.status == STARTED:
                    # update params
                    pass
                
                # *FixationShape_1* updates
                
                # if FixationShape_1 is starting this frame...
                if FixationShape_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationShape_1.frameNStart = frameN  # exact frame index
                    FixationShape_1.tStart = t  # local t and not account for scr refresh
                    FixationShape_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationShape_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationShape_1.started')
                    # update status
                    FixationShape_1.status = STARTED
                    FixationShape_1.setAutoDraw(True)
                
                # if FixationShape_1 is active this frame...
                if FixationShape_1.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Fixation,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Fixation" ---
            for thisComponent in Fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Fixation
            Fixation.tStop = globalClock.getTime(format='float')
            Fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Fixation.stopped', Fixation.tStop)
            # the Routine "Fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Stimulus" ---
            # create an object to store info about Routine Stimulus
            Stimulus = data.Routine(
                name='Stimulus',
                components=[CentralExclusionZone_3, FixationShape_2, GetResponse],
            )
            Stimulus.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_3
            win.color = [0, 0, 0] # keep grey background
            # Run 'Begin Routine' code from UpdateVariablesAndStaircase
            # Update trials variables (from globals)
            for variable_name in trial_variables.keys():
                if variable_name != "response":
                    trial_variables[variable_name] = globals().get(variable_name, None)
            
            # Infer next horizontal displacement
            if trial_variables["state"] == "test": # staircase during testing trials
                key = get_key(trial_variables["n_row"], trial_variables["orientation_degrees"])
                STAIRCASE = staircases[key] # overwrite from previous trial (and "none" initially)
                STIMULUS_DICT = STAIRCASE.next_stim
                trial_variables["displacement_cols"] = STIMULUS_DICT["intensity"]
                if trial_variables["orientation_degrees"] >= 90: # one staircase for symmetric oriented textures
                    trial_variables["displacement_cols"] = - trial_variables["displacement_cols"]
            # Run 'Begin Routine' code from DisplayStimulus
            win.color = [0, 0, 0] # keep grey background
            
            # randomize location of Gabor patch centers
            jitter = rng.uniform(-jitter_amount, jitter_amount, 
                                       size=(2, trial_variables["n_row"], n_col, 2))
            xys_3D = base_xys[trial_variables["n_row"]] + jitter
            
            # rotate to set orientation
            oris_3D = base_oris[trial_variables["n_row"]] + trial_variables["orientation_degrees"]
            
            # displace upper texture horizontally
            xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
            
            # set eccentricity of texture
            xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"] 
            xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"]
            
            # set and update stimulus
            gabor_array = gabor_arrays[trial_variables["n_row"]]
            gabor_array.xys = xys_3D.reshape(-1, 2)
            gabor_array.oris = oris_3D.flatten()
            # Run 'Begin Routine' code from DisplayMask
            # Set mask
            mask_array = mask_arrays[trial_variables["n_row"]]
            
            # Copy stimulus locations to avoid confusing salient shifts after stimulus
            xys_3D = base_xys[trial_variables["n_row"]] + jitter
            xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
            xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"]
            xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"] 
            mask_array.xys = xys_3D.reshape(-1, 2)
            
            # But scramble orientation and thus texture
            mask_array.oris = rng.uniform(0, 180, size=2 * trial_variables["n_row"] * n_col)
            CentralExclusionZone_3.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            FixationShape_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
            # create starting attributes for GetResponse
            GetResponse.keys = []
            GetResponse.rt = []
            _GetResponse_allKeys = []
            # store start times for Stimulus
            Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Stimulus.tStart = globalClock.getTime(format='float')
            Stimulus.status = STARTED
            thisExp.addData('Stimulus.started', Stimulus.tStart)
            Stimulus.maxDuration = None
            # keep track of which components have finished
            StimulusComponents = Stimulus.components
            for thisComponent in Stimulus.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Stimulus" ---
            Stimulus.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTraining_trial, 'status') and thisTraining_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from DisplayStimulus
                if frameN < trial_variables["stimulus_frames"]:
                    gabor_array.draw()
                # Run 'Each Frame' code from DisplayMask
                if frameN >= trial_variables["stimulus_frames"]:
                    mask_array.draw()
                
                # *CentralExclusionZone_3* updates
                
                # if CentralExclusionZone_3 is starting this frame...
                if CentralExclusionZone_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    CentralExclusionZone_3.frameNStart = frameN  # exact frame index
                    CentralExclusionZone_3.tStart = t  # local t and not account for scr refresh
                    CentralExclusionZone_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(CentralExclusionZone_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CentralExclusionZone_3.started')
                    # update status
                    CentralExclusionZone_3.status = STARTED
                    CentralExclusionZone_3.setAutoDraw(True)
                
                # if CentralExclusionZone_3 is active this frame...
                if CentralExclusionZone_3.status == STARTED:
                    # update params
                    pass
                
                # *FixationShape_2* updates
                
                # if FixationShape_2 is starting this frame...
                if FixationShape_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationShape_2.frameNStart = frameN  # exact frame index
                    FixationShape_2.tStart = t  # local t and not account for scr refresh
                    FixationShape_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationShape_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationShape_2.started')
                    # update status
                    FixationShape_2.status = STARTED
                    FixationShape_2.setAutoDraw(True)
                
                # if FixationShape_2 is active this frame...
                if FixationShape_2.status == STARTED:
                    # update params
                    pass
                
                # *GetResponse* updates
                waitOnFlip = False
                
                # if GetResponse is starting this frame...
                if GetResponse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    GetResponse.frameNStart = frameN  # exact frame index
                    GetResponse.tStart = t  # local t and not account for scr refresh
                    GetResponse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(GetResponse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'GetResponse.started')
                    # update status
                    GetResponse.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(GetResponse.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(GetResponse.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if GetResponse.status == STARTED and not waitOnFlip:
                    theseKeys = GetResponse.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _GetResponse_allKeys.extend(theseKeys)
                    if len(_GetResponse_allKeys):
                        GetResponse.keys = _GetResponse_allKeys[-1].name  # just the last key pressed
                        GetResponse.rt = _GetResponse_allKeys[-1].rt
                        GetResponse.duration = _GetResponse_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from SimulateResponse
                if expInfo.get('monkey_mode', False):
                    # simulate response if no key was pressed yet
                    if not GetResponse.keys:
                        # wait realistrically
                        if t > RT_monkey:  
                            simulated_key = simulate_response(
                                trial_variables["displacement_cols"], 
                                mu=PSE_monkey, 
                                sigma=STD_monkey
                            )
                            
                            GetResponse.keys = simulated_key
                            GetResponse.rt = RT_monkey
                            GetResponse.duration = None
                            continueRoutine = False  # force routine end
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Stimulus,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Stimulus.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Stimulus.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Stimulus" ---
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Stimulus
            Stimulus.tStop = globalClock.getTime(format='float')
            Stimulus.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Stimulus.stopped', Stimulus.tStop)
            # Run 'End Routine' code from UpdateVariablesAndStaircase
            # Update staircase
            if trial_variables["state"] == "test":
                # staircase: positive == towards the upper texture parallely oriented to border
                if trial_variables["orientation_degrees"] < 90: # right upper texture more parallel to border
                    response = 1 if (GetResponse.keys == "right") else 0
                elif trial_variables["orientation_degrees"] >= 90: # left upper texture more parallel to border
                    response = 1 if (GetResponse.keys == "left") else 0
                STAIRCASE.update(stim=STIMULUS_DICT, outcome={"response": response}) # variable "staircase" and "stimulus_dict" already selected at begin routine
            # Run 'End Routine' code from DisplayMask
            win.color = [0, 0, 0] # keep grey background
            # check responses
            if GetResponse.keys in ['', [], None]:  # No response was made
                GetResponse.keys = None
            training_trials.addData('GetResponse.keys',GetResponse.keys)
            if GetResponse.keys != None:  # we had a response
                training_trials.addData('GetResponse.rt', GetResponse.rt)
                training_trials.addData('GetResponse.duration', GetResponse.duration)
            # Run 'End Routine' code from SaveTrialData
            # save trial to trial data 
            # possibly redundant for non-staircase loops
            trial_variables["response"] = GetResponse.keys
            for variable_name, variable_value in trial_variables.items():
                thisExp.addData(variable_name, variable_value) 
            # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Blank" ---
            # create an object to store info about Routine Blank
            Blank = data.Routine(
                name='Blank',
                components=[],
            )
            Blank.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from SetBackground_4
            win.clearBuffer() 
            win.color = [0, 0, 0] # keep grey background
            # store start times for Blank
            Blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Blank.tStart = globalClock.getTime(format='float')
            Blank.status = STARTED
            thisExp.addData('Blank.started', Blank.tStart)
            # keep track of which components have finished
            BlankComponents = Blank.components
            for thisComponent in Blank.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Blank" ---
            Blank.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTraining_trial, 'status') and thisTraining_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= blank_frames:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Blank,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Blank.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Blank.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Blank" ---
            for thisComponent in Blank.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Blank
            Blank.tStop = globalClock.getTime(format='float')
            Blank.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Blank.stopped', Blank.tStop)
            # Run 'End Routine' code from CheckTimeAndTrialNumber
            # check clock
            if track_time and trial_variables["state"] == "test":
                elapsed_time_s = time.time() - practice_start_time_s
                if elapsed_time_s >= MAX_TIME_s:
                    # log
                    message = f"Time limit reached: {elapsed_time_s/60:.1f} minutes elapsed after {testing_trials.thisN + 1} testing trials"
                    print(message)
                    logging.info(message)
                    logging.exp(message)
                    
                    # Add summary data
                    thisExp.addData('time_limit_min', MAX_TIME_s / 60)
                    thisExp.addData('tracked_time_min', elapsed_time_s / 60)
                    thisExp.addData('time_limit_was_reached', True)
                    
                    # finish experiment
                    testing_trials.finished = True
            
            # check trial number
            if trial_variables["state"] == "train": # for some valid ExpInfo.yaml settings, there are more training conditions then nTrials_train
                if training_trials.thisN + 1 >= nTrials_train: # + 1 since thisN starts at 0
                    training_trials.finished = True
            elif not track_time and trial_variables["state"] == "test": # if not track time, assure right number of trials despite of rounding issues
                if testing_trials.thisN + 1 >= nTrials_test: # + 1 since thisN starts at 0
                    testing_trials.finished = True
            # the Routine "Blank" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTraining_trial as finished
            if hasattr(thisTraining_trial, 'status'):
                thisTraining_trial.status = FINISHED
            # if awaiting a pause, pause now
            if training_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                training_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed nReps_train repeats of 'training_trials'
        training_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if training_trials.trialList in ([], [None], None):
            params = []
        else:
            params = training_trials.trialList[0].keys()
        # save data for this loop
        training_trials.saveAsExcel(filename + '.xlsx', sheetName='training_trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        training_trials.saveAsText(filename + '_training_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "Dialogue" ---
        # create an object to store info about Routine Dialogue
        Dialogue = data.Routine(
            name='Dialogue',
            components=[RepeatOrContinueText, RepeatOrContinueButton],
        )
        Dialogue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for RepeatOrContinueButton
        RepeatOrContinueButton.keys = []
        RepeatOrContinueButton.rt = []
        _RepeatOrContinueButton_allKeys = []
        # store start times for Dialogue
        Dialogue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Dialogue.tStart = globalClock.getTime(format='float')
        Dialogue.status = STARTED
        thisExp.addData('Dialogue.started', Dialogue.tStart)
        Dialogue.maxDuration = None
        # keep track of which components have finished
        DialogueComponents = Dialogue.components
        for thisComponent in Dialogue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Dialogue" ---
        Dialogue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTraining, 'status') and thisTraining.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *RepeatOrContinueText* updates
            
            # if RepeatOrContinueText is starting this frame...
            if RepeatOrContinueText.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                RepeatOrContinueText.frameNStart = frameN  # exact frame index
                RepeatOrContinueText.tStart = t  # local t and not account for scr refresh
                RepeatOrContinueText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RepeatOrContinueText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RepeatOrContinueText.started')
                # update status
                RepeatOrContinueText.status = STARTED
                RepeatOrContinueText.setAutoDraw(True)
            
            # if RepeatOrContinueText is active this frame...
            if RepeatOrContinueText.status == STARTED:
                # update params
                pass
            
            # *RepeatOrContinueButton* updates
            waitOnFlip = False
            
            # if RepeatOrContinueButton is starting this frame...
            if RepeatOrContinueButton.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                RepeatOrContinueButton.frameNStart = frameN  # exact frame index
                RepeatOrContinueButton.tStart = t  # local t and not account for scr refresh
                RepeatOrContinueButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RepeatOrContinueButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RepeatOrContinueButton.started')
                # update status
                RepeatOrContinueButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(RepeatOrContinueButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(RepeatOrContinueButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if RepeatOrContinueButton.status == STARTED and not waitOnFlip:
                theseKeys = RepeatOrContinueButton.getKeys(keyList=['r','c'], ignoreKeys=["escape"], waitRelease=False)
                _RepeatOrContinueButton_allKeys.extend(theseKeys)
                if len(_RepeatOrContinueButton_allKeys):
                    RepeatOrContinueButton.keys = _RepeatOrContinueButton_allKeys[-1].name  # just the last key pressed
                    RepeatOrContinueButton.rt = _RepeatOrContinueButton_allKeys[-1].rt
                    RepeatOrContinueButton.duration = _RepeatOrContinueButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Dialogue,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Dialogue.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Dialogue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Dialogue" ---
        for thisComponent in Dialogue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Dialogue
        Dialogue.tStop = globalClock.getTime(format='float')
        Dialogue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Dialogue.stopped', Dialogue.tStop)
        # check responses
        if RepeatOrContinueButton.keys in ['', [], None]:  # No response was made
            RepeatOrContinueButton.keys = None
        training.addData('RepeatOrContinueButton.keys',RepeatOrContinueButton.keys)
        if RepeatOrContinueButton.keys != None:  # we had a response
            training.addData('RepeatOrContinueButton.rt', RepeatOrContinueButton.rt)
            training.addData('RepeatOrContinueButton.duration', RepeatOrContinueButton.duration)
        # Run 'End Routine' code from FinishLoop
        if RepeatOrContinueButton.keys == 'c':
            if trial_variables["state"] == 'instruct': # stop instructions
                instruction.finished = True
            elif trial_variables["state"] == 'train': # stop training / practice
                training.finished = True
        # the Routine "Dialogue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTraining as finished
        if hasattr(thisTraining, 'status'):
            thisTraining.status = FINISHED
        # if awaiting a pause, pause now
        if training.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            training.status = STARTED
    # completed 999 if expInfo['skip_training'] == False else 0 repeats of 'training'
    training.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    testing_trials = data.TrialHandler2(
        name='testing_trials',
        nReps=nReps_test if expInfo["skip_testing"] == False else 0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions_temp_test.csv'), 
        seed=seed, 
    )
    thisExp.addLoop(testing_trials)  # add the loop to the experiment
    thisTesting_trial = testing_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTesting_trial.rgb)
    if thisTesting_trial != None:
        for paramName in thisTesting_trial:
            globals()[paramName] = thisTesting_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTesting_trial in testing_trials:
        testing_trials.status = STARTED
        if hasattr(thisTesting_trial, 'status'):
            thisTesting_trial.status = STARTED
        currentLoop = testing_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTesting_trial.rgb)
        if thisTesting_trial != None:
            for paramName in thisTesting_trial:
                globals()[paramName] = thisTesting_trial[paramName]
        
        # --- Prepare to start Routine "Wait" ---
        # create an object to store info about Routine Wait
        Wait = data.Routine(
            name='Wait',
            components=[AnyButton, CentralExclusionZone_1, PressAnyButtonForNextTrialText],
        )
        Wait.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from SetBackground
        win.color = [0, 0, 0] # assure grey background
        # Run 'Begin Routine' code from PrintTrialNumber
        TRIAL_COUNT += 1
        print(f"{state} trial {TRIAL_COUNT}") # print to note down and refind errors
        thisExp.addData('global.thisN', TRIAL_COUNT) # save to compare with noted down global trial number
        # create starting attributes for AnyButton
        AnyButton.keys = []
        AnyButton.rt = []
        _AnyButton_allKeys = []
        CentralExclusionZone_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        PressAnyButtonForNextTrialText.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        PressAnyButtonForNextTrialText.setText("Press any button\nfor %d. trial" % (TRIAL_COUNT))
        # store start times for Wait
        Wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Wait.tStart = globalClock.getTime(format='float')
        Wait.status = STARTED
        thisExp.addData('Wait.started', Wait.tStart)
        Wait.maxDuration = None
        # keep track of which components have finished
        WaitComponents = Wait.components
        for thisComponent in Wait.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Wait" ---
        Wait.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTesting_trial, 'status') and thisTesting_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *AnyButton* updates
            waitOnFlip = False
            
            # if AnyButton is starting this frame...
            if AnyButton.status == NOT_STARTED and frameN >= 10:
                # keep track of start time/frame for later
                AnyButton.frameNStart = frameN  # exact frame index
                AnyButton.tStart = t  # local t and not account for scr refresh
                AnyButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(AnyButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'AnyButton.started')
                # update status
                AnyButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(AnyButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(AnyButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if AnyButton.status == STARTED and not waitOnFlip:
                theseKeys = AnyButton.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _AnyButton_allKeys.extend(theseKeys)
                if len(_AnyButton_allKeys):
                    AnyButton.keys = _AnyButton_allKeys[-1].name  # just the last key pressed
                    AnyButton.rt = _AnyButton_allKeys[-1].rt
                    AnyButton.duration = _AnyButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *CentralExclusionZone_1* updates
            
            # if CentralExclusionZone_1 is starting this frame...
            if CentralExclusionZone_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                CentralExclusionZone_1.frameNStart = frameN  # exact frame index
                CentralExclusionZone_1.tStart = t  # local t and not account for scr refresh
                CentralExclusionZone_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(CentralExclusionZone_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'CentralExclusionZone_1.started')
                # update status
                CentralExclusionZone_1.status = STARTED
                CentralExclusionZone_1.setAutoDraw(True)
            
            # if CentralExclusionZone_1 is active this frame...
            if CentralExclusionZone_1.status == STARTED:
                # update params
                pass
            
            # *PressAnyButtonForNextTrialText* updates
            
            # if PressAnyButtonForNextTrialText is starting this frame...
            if PressAnyButtonForNextTrialText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                PressAnyButtonForNextTrialText.frameNStart = frameN  # exact frame index
                PressAnyButtonForNextTrialText.tStart = t  # local t and not account for scr refresh
                PressAnyButtonForNextTrialText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PressAnyButtonForNextTrialText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PressAnyButtonForNextTrialText.started')
                # update status
                PressAnyButtonForNextTrialText.status = STARTED
                PressAnyButtonForNextTrialText.setAutoDraw(True)
            
            # if PressAnyButtonForNextTrialText is active this frame...
            if PressAnyButtonForNextTrialText.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from SimulateButtonPress
            if expInfo.get('monkey_mode', False):
                # simulate response if no key was pressed yet
                if not AnyButton.keys:
                    if t > RT_monkey:
                        AnyButton.keys = "enter"
                        AnyButton.rt = RT_monkey
                        AnyButton.duration = None
                        continueRoutine = False 
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Wait,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Wait.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Wait.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Wait" ---
        for thisComponent in Wait.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Wait
        Wait.tStop = globalClock.getTime(format='float')
        Wait.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Wait.stopped', Wait.tStop)
        # check responses
        if AnyButton.keys in ['', [], None]:  # No response was made
            AnyButton.keys = None
        testing_trials.addData('AnyButton.keys',AnyButton.keys)
        if AnyButton.keys != None:  # we had a response
            testing_trials.addData('AnyButton.rt', AnyButton.rt)
            testing_trials.addData('AnyButton.duration', AnyButton.duration)
        # the Routine "Wait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Fixation" ---
        # create an object to store info about Routine Fixation
        Fixation = data.Routine(
            name='Fixation',
            components=[CentralExclusionZone_2, FixationShape_1],
        )
        Fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from SetBackground_2
        win.color = [0, 0, 0] # keep grey background
        CentralExclusionZone_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        FixationShape_1.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        # store start times for Fixation
        Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fixation.tStart = globalClock.getTime(format='float')
        Fixation.status = STARTED
        thisExp.addData('Fixation.started', Fixation.tStart)
        # keep track of which components have finished
        FixationComponents = Fixation.components
        for thisComponent in Fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation" ---
        Fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTesting_trial, 'status') and thisTesting_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on frames since Routine start)
            if frameN >= fixation_frames:
                continueRoutine = False
            
            # *CentralExclusionZone_2* updates
            
            # if CentralExclusionZone_2 is starting this frame...
            if CentralExclusionZone_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                CentralExclusionZone_2.frameNStart = frameN  # exact frame index
                CentralExclusionZone_2.tStart = t  # local t and not account for scr refresh
                CentralExclusionZone_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(CentralExclusionZone_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'CentralExclusionZone_2.started')
                # update status
                CentralExclusionZone_2.status = STARTED
                CentralExclusionZone_2.setAutoDraw(True)
            
            # if CentralExclusionZone_2 is active this frame...
            if CentralExclusionZone_2.status == STARTED:
                # update params
                pass
            
            # *FixationShape_1* updates
            
            # if FixationShape_1 is starting this frame...
            if FixationShape_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                FixationShape_1.frameNStart = frameN  # exact frame index
                FixationShape_1.tStart = t  # local t and not account for scr refresh
                FixationShape_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationShape_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationShape_1.started')
                # update status
                FixationShape_1.status = STARTED
                FixationShape_1.setAutoDraw(True)
            
            # if FixationShape_1 is active this frame...
            if FixationShape_1.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Fixation,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation" ---
        for thisComponent in Fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fixation
        Fixation.tStop = globalClock.getTime(format='float')
        Fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fixation.stopped', Fixation.tStop)
        # the Routine "Fixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[CentralExclusionZone_3, FixationShape_2, GetResponse],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from SetBackground_3
        win.color = [0, 0, 0] # keep grey background
        # Run 'Begin Routine' code from UpdateVariablesAndStaircase
        # Update trials variables (from globals)
        for variable_name in trial_variables.keys():
            if variable_name != "response":
                trial_variables[variable_name] = globals().get(variable_name, None)
        
        # Infer next horizontal displacement
        if trial_variables["state"] == "test": # staircase during testing trials
            key = get_key(trial_variables["n_row"], trial_variables["orientation_degrees"])
            STAIRCASE = staircases[key] # overwrite from previous trial (and "none" initially)
            STIMULUS_DICT = STAIRCASE.next_stim
            trial_variables["displacement_cols"] = STIMULUS_DICT["intensity"]
            if trial_variables["orientation_degrees"] >= 90: # one staircase for symmetric oriented textures
                trial_variables["displacement_cols"] = - trial_variables["displacement_cols"]
        # Run 'Begin Routine' code from DisplayStimulus
        win.color = [0, 0, 0] # keep grey background
        
        # randomize location of Gabor patch centers
        jitter = rng.uniform(-jitter_amount, jitter_amount, 
                                   size=(2, trial_variables["n_row"], n_col, 2))
        xys_3D = base_xys[trial_variables["n_row"]] + jitter
        
        # rotate to set orientation
        oris_3D = base_oris[trial_variables["n_row"]] + trial_variables["orientation_degrees"]
        
        # displace upper texture horizontally
        xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
        
        # set eccentricity of texture
        xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"] 
        xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"]
        
        # set and update stimulus
        gabor_array = gabor_arrays[trial_variables["n_row"]]
        gabor_array.xys = xys_3D.reshape(-1, 2)
        gabor_array.oris = oris_3D.flatten()
        # Run 'Begin Routine' code from DisplayMask
        # Set mask
        mask_array = mask_arrays[trial_variables["n_row"]]
        
        # Copy stimulus locations to avoid confusing salient shifts after stimulus
        xys_3D = base_xys[trial_variables["n_row"]] + jitter
        xys_3D[0, :, :, 0] += spacing * trial_variables["displacement_cols"]
        xys_3D[:, :, :, 0] += trial_variables["x_eccentricity_degrees"]
        xys_3D[:, :, :, 1] += trial_variables["y_eccentricity_degrees"] 
        mask_array.xys = xys_3D.reshape(-1, 2)
        
        # But scramble orientation and thus texture
        mask_array.oris = rng.uniform(0, 180, size=2 * trial_variables["n_row"] * n_col)
        CentralExclusionZone_3.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        FixationShape_2.setPos((-x_eccentricity_degrees, -y_eccentricity_degrees))
        # create starting attributes for GetResponse
        GetResponse.keys = []
        GetResponse.rt = []
        _GetResponse_allKeys = []
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTesting_trial, 'status') and thisTesting_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from DisplayStimulus
            if frameN < trial_variables["stimulus_frames"]:
                gabor_array.draw()
            # Run 'Each Frame' code from DisplayMask
            if frameN >= trial_variables["stimulus_frames"]:
                mask_array.draw()
            
            # *CentralExclusionZone_3* updates
            
            # if CentralExclusionZone_3 is starting this frame...
            if CentralExclusionZone_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                CentralExclusionZone_3.frameNStart = frameN  # exact frame index
                CentralExclusionZone_3.tStart = t  # local t and not account for scr refresh
                CentralExclusionZone_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(CentralExclusionZone_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'CentralExclusionZone_3.started')
                # update status
                CentralExclusionZone_3.status = STARTED
                CentralExclusionZone_3.setAutoDraw(True)
            
            # if CentralExclusionZone_3 is active this frame...
            if CentralExclusionZone_3.status == STARTED:
                # update params
                pass
            
            # *FixationShape_2* updates
            
            # if FixationShape_2 is starting this frame...
            if FixationShape_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationShape_2.frameNStart = frameN  # exact frame index
                FixationShape_2.tStart = t  # local t and not account for scr refresh
                FixationShape_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationShape_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationShape_2.started')
                # update status
                FixationShape_2.status = STARTED
                FixationShape_2.setAutoDraw(True)
            
            # if FixationShape_2 is active this frame...
            if FixationShape_2.status == STARTED:
                # update params
                pass
            
            # *GetResponse* updates
            waitOnFlip = False
            
            # if GetResponse is starting this frame...
            if GetResponse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                GetResponse.frameNStart = frameN  # exact frame index
                GetResponse.tStart = t  # local t and not account for scr refresh
                GetResponse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(GetResponse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'GetResponse.started')
                # update status
                GetResponse.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(GetResponse.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(GetResponse.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if GetResponse.status == STARTED and not waitOnFlip:
                theseKeys = GetResponse.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _GetResponse_allKeys.extend(theseKeys)
                if len(_GetResponse_allKeys):
                    GetResponse.keys = _GetResponse_allKeys[-1].name  # just the last key pressed
                    GetResponse.rt = _GetResponse_allKeys[-1].rt
                    GetResponse.duration = _GetResponse_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from SimulateResponse
            if expInfo.get('monkey_mode', False):
                # simulate response if no key was pressed yet
                if not GetResponse.keys:
                    # wait realistrically
                    if t > RT_monkey:  
                        simulated_key = simulate_response(
                            trial_variables["displacement_cols"], 
                            mu=PSE_monkey, 
                            sigma=STD_monkey
                        )
                        
                        GetResponse.keys = simulated_key
                        GetResponse.rt = RT_monkey
                        GetResponse.duration = None
                        continueRoutine = False  # force routine end
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Stimulus,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # Run 'End Routine' code from UpdateVariablesAndStaircase
        # Update staircase
        if trial_variables["state"] == "test":
            # staircase: positive == towards the upper texture parallely oriented to border
            if trial_variables["orientation_degrees"] < 90: # right upper texture more parallel to border
                response = 1 if (GetResponse.keys == "right") else 0
            elif trial_variables["orientation_degrees"] >= 90: # left upper texture more parallel to border
                response = 1 if (GetResponse.keys == "left") else 0
            STAIRCASE.update(stim=STIMULUS_DICT, outcome={"response": response}) # variable "staircase" and "stimulus_dict" already selected at begin routine
        # Run 'End Routine' code from DisplayMask
        win.color = [0, 0, 0] # keep grey background
        # check responses
        if GetResponse.keys in ['', [], None]:  # No response was made
            GetResponse.keys = None
        testing_trials.addData('GetResponse.keys',GetResponse.keys)
        if GetResponse.keys != None:  # we had a response
            testing_trials.addData('GetResponse.rt', GetResponse.rt)
            testing_trials.addData('GetResponse.duration', GetResponse.duration)
        # Run 'End Routine' code from SaveTrialData
        # save trial to trial data 
        # possibly redundant for non-staircase loops
        trial_variables["response"] = GetResponse.keys
        for variable_name, variable_value in trial_variables.items():
            thisExp.addData(variable_name, variable_value) 
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Blank" ---
        # create an object to store info about Routine Blank
        Blank = data.Routine(
            name='Blank',
            components=[],
        )
        Blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from SetBackground_4
        win.clearBuffer() 
        win.color = [0, 0, 0] # keep grey background
        # store start times for Blank
        Blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Blank.tStart = globalClock.getTime(format='float')
        Blank.status = STARTED
        thisExp.addData('Blank.started', Blank.tStart)
        # keep track of which components have finished
        BlankComponents = Blank.components
        for thisComponent in Blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Blank" ---
        Blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTesting_trial, 'status') and thisTesting_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on frames since Routine start)
            if frameN >= blank_frames:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Blank,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Blank" ---
        for thisComponent in Blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Blank
        Blank.tStop = globalClock.getTime(format='float')
        Blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Blank.stopped', Blank.tStop)
        # Run 'End Routine' code from CheckTimeAndTrialNumber
        # check clock
        if track_time and trial_variables["state"] == "test":
            elapsed_time_s = time.time() - practice_start_time_s
            if elapsed_time_s >= MAX_TIME_s:
                # log
                message = f"Time limit reached: {elapsed_time_s/60:.1f} minutes elapsed after {testing_trials.thisN + 1} testing trials"
                print(message)
                logging.info(message)
                logging.exp(message)
                
                # Add summary data
                thisExp.addData('time_limit_min', MAX_TIME_s / 60)
                thisExp.addData('tracked_time_min', elapsed_time_s / 60)
                thisExp.addData('time_limit_was_reached', True)
                
                # finish experiment
                testing_trials.finished = True
        
        # check trial number
        if trial_variables["state"] == "train": # for some valid ExpInfo.yaml settings, there are more training conditions then nTrials_train
            if training_trials.thisN + 1 >= nTrials_train: # + 1 since thisN starts at 0
                training_trials.finished = True
        elif not track_time and trial_variables["state"] == "test": # if not track time, assure right number of trials despite of rounding issues
            if testing_trials.thisN + 1 >= nTrials_test: # + 1 since thisN starts at 0
                testing_trials.finished = True
        # the Routine "Blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTesting_trial as finished
        if hasattr(thisTesting_trial, 'status'):
            thisTesting_trial.status = FINISHED
        # if awaiting a pause, pause now
        if testing_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            testing_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed nReps_test if expInfo["skip_testing"] == False else 0 repeats of 'testing_trials'
    testing_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if testing_trials.trialList in ([], [None], None):
        params = []
    else:
        params = testing_trials.trialList[0].keys()
    # save data for this loop
    testing_trials.saveAsExcel(filename + '.xlsx', sheetName='testing_trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    testing_trials.saveAsText(filename + '_testing_trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "Break" ---
    # create an object to store info about Routine Break
    Break = data.Routine(
        name='Break',
        components=[TakeABreakText, AnyButton_2],
    )
    Break.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for AnyButton_2
    AnyButton_2.keys = []
    AnyButton_2.rt = []
    _AnyButton_2_allKeys = []
    # store start times for Break
    Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Break.tStart = globalClock.getTime(format='float')
    Break.status = STARTED
    thisExp.addData('Break.started', Break.tStart)
    Break.maxDuration = None
    # keep track of which components have finished
    BreakComponents = Break.components
    for thisComponent in Break.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Break" ---
    Break.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *TakeABreakText* updates
        
        # if TakeABreakText is starting this frame...
        if TakeABreakText.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            TakeABreakText.frameNStart = frameN  # exact frame index
            TakeABreakText.tStart = t  # local t and not account for scr refresh
            TakeABreakText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(TakeABreakText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'TakeABreakText.started')
            # update status
            TakeABreakText.status = STARTED
            TakeABreakText.setAutoDraw(True)
        
        # if TakeABreakText is active this frame...
        if TakeABreakText.status == STARTED:
            # update params
            pass
        
        # *AnyButton_2* updates
        waitOnFlip = False
        
        # if AnyButton_2 is starting this frame...
        if AnyButton_2.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            AnyButton_2.frameNStart = frameN  # exact frame index
            AnyButton_2.tStart = t  # local t and not account for scr refresh
            AnyButton_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(AnyButton_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'AnyButton_2.started')
            # update status
            AnyButton_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(AnyButton_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(AnyButton_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if AnyButton_2.status == STARTED and not waitOnFlip:
            theseKeys = AnyButton_2.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _AnyButton_2_allKeys.extend(theseKeys)
            if len(_AnyButton_2_allKeys):
                AnyButton_2.keys = _AnyButton_2_allKeys[-1].name  # just the last key pressed
                AnyButton_2.rt = _AnyButton_2_allKeys[-1].rt
                AnyButton_2.duration = _AnyButton_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Break,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Break.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Break.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Break" ---
    for thisComponent in Break.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Break
    Break.tStop = globalClock.getTime(format='float')
    Break.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Break.stopped', Break.tStop)
    # check responses
    if AnyButton_2.keys in ['', [], None]:  # No response was made
        AnyButton_2.keys = None
    thisExp.addData('AnyButton_2.keys',AnyButton_2.keys)
    if AnyButton_2.keys != None:  # we had a response
        thisExp.addData('AnyButton_2.rt', AnyButton_2.rt)
        thisExp.addData('AnyButton_2.duration', AnyButton_2.duration)
    thisExp.nextEntry()
    # the Routine "Break" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "QandASession" ---
    # create an object to store info about Routine QandASession
    QandASession = data.Routine(
        name='QandASession',
        components=[QandAForm, FinishButton],
    )
    QandASession.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from ShowMouse
    win.mouseVisible = True
    # reset FinishButton to account for continued clicks & clear times on/off
    FinishButton.reset()
    # store start times for QandASession
    QandASession.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    QandASession.tStart = globalClock.getTime(format='float')
    QandASession.status = STARTED
    thisExp.addData('QandASession.started', QandASession.tStart)
    QandASession.maxDuration = None
    # keep track of which components have finished
    QandASessionComponents = QandASession.components
    for thisComponent in QandASession.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "QandASession" ---
    QandASession.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *QandAForm* updates
        
        # if QandAForm is starting this frame...
        if QandAForm.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            QandAForm.frameNStart = frameN  # exact frame index
            QandAForm.tStart = t  # local t and not account for scr refresh
            QandAForm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(QandAForm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'QandAForm.started')
            # update status
            QandAForm.status = STARTED
            QandAForm.setAutoDraw(True)
        
        # if QandAForm is active this frame...
        if QandAForm.status == STARTED:
            # update params
            pass
        # *FinishButton* updates
        
        # if FinishButton is starting this frame...
        if FinishButton.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
            # keep track of start time/frame for later
            FinishButton.frameNStart = frameN  # exact frame index
            FinishButton.tStart = t  # local t and not account for scr refresh
            FinishButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FinishButton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'FinishButton.started')
            # update status
            FinishButton.status = STARTED
            win.callOnFlip(FinishButton.buttonClock.reset)
            FinishButton.setAutoDraw(True)
        
        # if FinishButton is active this frame...
        if FinishButton.status == STARTED:
            # update params
            pass
            # check whether FinishButton has been pressed
            if FinishButton.isClicked:
                if not FinishButton.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    FinishButton.timesOn.append(FinishButton.buttonClock.getTime())
                    FinishButton.timesOff.append(FinishButton.buttonClock.getTime())
                elif len(FinishButton.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    FinishButton.timesOff[-1] = FinishButton.buttonClock.getTime()
                if not FinishButton.wasClicked:
                    # end routine when FinishButton is clicked
                    continueRoutine = False
                if not FinishButton.wasClicked:
                    # run callback code when FinishButton is clicked
                    pass
        # take note of whether FinishButton was clicked, so that next frame we know if clicks are new
        FinishButton.wasClicked = FinishButton.isClicked and FinishButton.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=QandASession,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            QandASession.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in QandASession.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "QandASession" ---
    for thisComponent in QandASession.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for QandASession
    QandASession.tStop = globalClock.getTime(format='float')
    QandASession.tStopRefresh = tThisFlipGlobal
    thisExp.addData('QandASession.stopped', QandASession.tStop)
    # Run 'End Routine' code from ShowMouse
    win.mouseVisible = False
    QandAForm.addDataToExp(thisExp, 'rows')
    QandAForm.autodraw = False
    thisExp.addData('FinishButton.numClicks', FinishButton.numClicks)
    if FinishButton.numClicks:
       thisExp.addData('FinishButton.timesOn', FinishButton.timesOn)
       thisExp.addData('FinishButton.timesOff', FinishButton.timesOff)
    else:
       thisExp.addData('FinishButton.timesOn', "")
       thisExp.addData('FinishButton.timesOff', "")
    thisExp.nextEntry()
    # the Routine "QandASession" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "SaveData" ---
    # create an object to store info about Routine SaveData
    SaveData = data.Routine(
        name='SaveData',
        components=[],
    )
    SaveData.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for SaveData
    SaveData.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SaveData.tStart = globalClock.getTime(format='float')
    SaveData.status = STARTED
    thisExp.addData('SaveData.started', SaveData.tStart)
    SaveData.maxDuration = None
    # keep track of which components have finished
    SaveDataComponents = SaveData.components
    for thisComponent in SaveData.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SaveData" ---
    SaveData.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SaveData,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            SaveData.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in SaveData.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SaveData" ---
    for thisComponent in SaveData.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SaveData
    SaveData.tStop = globalClock.getTime(format='float')
    SaveData.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SaveData.stopped', SaveData.tStop)
    thisExp.nextEntry()
    # the Routine "SaveData" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[ThankYou],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.1:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ThankYou* updates
        
        # if ThankYou is starting this frame...
        if ThankYou.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            ThankYou.frameNStart = frameN  # exact frame index
            ThankYou.tStart = t  # local t and not account for scr refresh
            ThankYou.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ThankYou, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ThankYou.started')
            # update status
            ThankYou.status = STARTED
            ThankYou.setAutoDraw(True)
        
        # if ThankYou is active this frame...
        if ThankYou.status == STARTED:
            # update params
            pass
        
        # if ThankYou is stopping this frame...
        if ThankYou.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ThankYou.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                ThankYou.tStop = t  # not accounting for scr refresh
                ThankYou.tStopRefresh = tThisFlipGlobal  # on global time
                ThankYou.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ThankYou.stopped')
                # update status
                ThankYou.status = FINISHED
                ThankYou.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            End.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if End.maxDurationReached:
        routineTimer.addTime(-End.maxDuration)
    elif End.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.100000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(enable=False)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
