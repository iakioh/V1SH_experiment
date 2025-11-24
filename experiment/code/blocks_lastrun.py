#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on November 19, 2025, at 17:38
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

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
from psychopy_monkeys import Monkey

# Run 'Before Experiment' code from TextureBorder
from psychopy import visual
import numpy as np
import yaml
import os
import math
import pandas as pd
import itertools

# Get Experimental Info
ExpInfo_path = os.path.join(os.path.dirname(__file__), '../ExpInfo.yaml')
with open(ExpInfo_path, 'r') as f:
    ExpInfo = yaml.safe_load(f)

StimInfo = ExpInfo["Stimulus"]
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'v0'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'age': '25',
    'gender': 'male',
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
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

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
    filename = u'../../data/personal/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\kai.rothe\\Documents\\V1SH_experiment\\experiment\\code\\blocks_lastrun.py',
        savePickle=True, saveWideText=False,
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
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
    if deviceManager.getDevice('SubjectResponse') is None:
        # initialise SubjectResponse
        SubjectResponse = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='SubjectResponse',
        )
    if deviceManager.getDevice('AnyButton') is None:
        # initialise AnyButton
        AnyButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='AnyButton',
        )
    if deviceManager.getDevice('WaitForEsc') is None:
        # initialise WaitForEsc
        WaitForEsc = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='WaitForEsc',
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
    
    # --- Initialize components for Routine "Instruction" ---
    
    # --- Initialize components for Routine "Fixation" ---
    FixationShape = visual.ShapeStim(
        win=win, name='FixationShape',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Stimulus" ---
    # Run 'Begin Experiment' code from TextureBorder
    # Set parameters
    nReps = ExpInfo["Procedure&Conditions"]["NumberOfRepetitions"]
    seed = ExpInfo["Procedure&Conditions"]["RandomSeed"]
    n_rows = StimInfo["Grid"]["Rows"]
    n_col = StimInfo["Grid"]["Columns"]
    vertical_seperation = StimInfo["VerticalSeperation_degrees"]
    frequency = StimInfo["GaborPatch"]["SpatialFrequency_cpd"] # cycles per degree
    mitchell_contrast = StimInfo["GaborPatch"]["MitchellContrast"] 
    duration = StimInfo["GaborPatch"]["Duration_s"]
    sigma = StimInfo["GaborPatch"]["Sigma_degrees"] # degrees
    spacing = StimInfo["GaborPatch"]["PatchSize_sigma"] * sigma  # degrees
    jitter_amount = StimInfo["Jitter"]["Range_sigma"] * sigma  # max jitter in degrees
    session_index = int(expInfo["session"]) - 1 # starts with "001"
    if len(StimInfo["GaborPatch"]["Orientation_degrees"]) <= session_index:
        raise TypeError("Number of orientations <= session number.")
    else:
        orientation = StimInfo["GaborPatch"]["Orientation_degrees"][session_index]
    
    # Create and save conditions sequence
    conditions = list(itertools.product(n_rows, StimInfo["HorizontalDisplacement_rows"]))
    conditions_df = pd.DataFrame(conditions, columns=['n_row', 'displacement_rows'])
    conditions_df['session'] = session_index 
    conditions_df['orientation_degrees'] = orientation  
    filename = 'conditions.csv'
    if os.path.exists(filename):
        # Append without header
        conditions_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create new file with header
        conditions_df.to_csv(filename, index=False)
    
    # Pre-compute timing
    refresh_rate = win.getActualFrameRate() # Hz
    stimulus_frames = math.floor(refresh_rate * duration)
    
    # Pre-Compute stimuli
    base_xys = {}
    gabor_arrays = {}
    for n_row_ in n_rows: # name n_row_ to avoid overwriting trial loop n_row condition
        # Pre-compute grid
        xs = np.linspace(-spacing * (n_col-1) / 2, spacing * (n_col-1) / 2, n_col)
        ys = np.linspace(0.0, spacing*(n_row_-1), n_row_)
        xys_upper = np.array([(x, y + vertical_seperation / 2) for y in ys for x in xs])
        xys_lower = np.array([(x, - y - vertical_seperation / 2) for y in ys for x in xs])
        xys = np.concatenate([xys_upper, xys_lower])
        base_xys[n_row_] = xys
    
        # Pre-compute orientations
        oris_upper = np.full((n_row_, n_col), orientation)
        oris_upper[:, n_col // 2 :] += 90
        oris_lower = np.full((n_row_, n_col), orientation)
        oris_lower[:, : n_col // 2] += 90
        oris = np.concatenate([oris_upper.flatten(), oris_lower.flatten()])
    
        # Pre-compute Gabor patches 
        gabor_array = visual.ElementArrayStim(
            win=win,
            units='deg',
            nElements= 2 * n_row_ * n_col,
            elementTex='sin',
            elementMask='gauss',
            xys=xys,
            sizes=spacing,
            sfs=frequency,
            oris=oris,
            contrs=mitchell_contrast
        )
        gabor_arrays[n_row_] = gabor_array
    SubjectResponse = keyboard.Keyboard(deviceName='SubjectResponse')
    DummyShape = visual.ShapeStim(
        win=win, name='DummyShape',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=0.0, depth=-2.0, interpolate=True)
    keyMonkey = Monkey(
        name='keyMonkey',
        comp=SubjectResponse,
    )
    
    # --- Initialize components for Routine "Wait" ---
    AnyButton = keyboard.Keyboard(deviceName='AnyButton')
    PressAnyButtonText = visual.TextStim(win=win, name='PressAnyButtonText',
        text='Press any button.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    keyMonkey_2 = Monkey(
        name='keyMonkey_2',
        comp=AnyButton,
    )
    
    # --- Initialize components for Routine "Break_3" ---
    TakeABreakText = visual.TextStim(win=win, name='TakeABreakText',
        text='Take a break!',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    WaitForEsc = keyboard.Keyboard(deviceName='WaitForEsc')
    
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
    
    # --- Prepare to start Routine "Instruction" ---
    # create an object to store info about Routine Instruction
    Instruction = data.Routine(
        name='Instruction',
        components=[],
    )
    Instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Instruction
    Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruction.tStart = globalClock.getTime(format='float')
    Instruction.status = STARTED
    thisExp.addData('Instruction.started', Instruction.tStart)
    Instruction.maxDuration = None
    # keep track of which components have finished
    InstructionComponents = Instruction.components
    for thisComponent in Instruction.components:
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
    
    # --- Run Routine "Instruction" ---
    Instruction.forceEnded = routineForceEnded = not continueRoutine
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
                currentRoutine=Instruction,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction" ---
    for thisComponent in Instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruction
    Instruction.tStop = globalClock.getTime(format='float')
    Instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruction.stopped', Instruction.tStop)
    thisExp.nextEntry()
    # the Routine "Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=nReps, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=seed, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Fixation" ---
        # create an object to store info about Routine Fixation
        Fixation = data.Routine(
            name='Fixation',
            components=[FixationShape],
        )
        Fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fixation
        Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fixation.tStart = globalClock.getTime(format='float')
        Fixation.status = STARTED
        thisExp.addData('Fixation.started', Fixation.tStart)
        Fixation.maxDuration = 0.6
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
        while continueRoutine and routineTimer.getTime() < 0.6:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > Fixation.maxDuration-frameTolerance:
                Fixation.maxDurationReached = True
                continueRoutine = False
            
            # *FixationShape* updates
            
            # if FixationShape is starting this frame...
            if FixationShape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationShape.frameNStart = frameN  # exact frame index
                FixationShape.tStart = t  # local t and not account for scr refresh
                FixationShape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationShape, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationShape.started')
                # update status
                FixationShape.status = STARTED
                FixationShape.setAutoDraw(True)
            
            # if FixationShape is active this frame...
            if FixationShape.status == STARTED:
                # update params
                pass
            
            # if FixationShape is stopping this frame...
            if FixationShape.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationShape.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationShape.tStop = t  # not accounting for scr refresh
                    FixationShape.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationShape.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationShape.stopped')
                    # update status
                    FixationShape.status = FINISHED
                    FixationShape.setAutoDraw(False)
            
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
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fixation.maxDurationReached:
            routineTimer.addTime(-Fixation.maxDuration)
        elif Fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.600000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[SubjectResponse, DummyShape, keyMonkey],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from TextureBorder
        # Decide for stimulus
        gabor_array = gabor_arrays[n_row]
        
        # Randomize locations
        jitter = np.random.uniform(-jitter_amount, jitter_amount, 
                                   size=(2 * n_row * n_col, 2))
        gabor_array.xys = base_xys[n_row] + jitter
        gabor_array.xys[:n_row * n_col] += spacing * displacement_rows 
        
        # Randomize phases
        # gabor_array.phases = np.random.uniform(0, 360, size=2 * n_rows * n_cols)
        # create starting attributes for SubjectResponse
        SubjectResponse.keys = []
        SubjectResponse.rt = []
        _SubjectResponse_allKeys = []
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from TextureBorder
            if frameN < stimulus_frames:
                gabor_array.draw()
            
            # *SubjectResponse* updates
            waitOnFlip = False
            
            # if SubjectResponse is starting this frame...
            if SubjectResponse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SubjectResponse.frameNStart = frameN  # exact frame index
                SubjectResponse.tStart = t  # local t and not account for scr refresh
                SubjectResponse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(SubjectResponse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'SubjectResponse.started')
                # update status
                SubjectResponse.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(SubjectResponse.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(SubjectResponse.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if SubjectResponse.status == STARTED and not waitOnFlip:
                theseKeys = SubjectResponse.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _SubjectResponse_allKeys.extend(theseKeys)
                if len(_SubjectResponse_allKeys):
                    SubjectResponse.keys = _SubjectResponse_allKeys[-1].name  # just the last key pressed
                    SubjectResponse.rt = _SubjectResponse_allKeys[-1].rt
                    SubjectResponse.duration = _SubjectResponse_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *DummyShape* updates
            
            # if DummyShape is starting this frame...
            if DummyShape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                DummyShape.frameNStart = frameN  # exact frame index
                DummyShape.tStart = t  # local t and not account for scr refresh
                DummyShape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(DummyShape, 'tStartRefresh')  # time at next scr refresh
                # update status
                DummyShape.status = STARTED
                DummyShape.setAutoDraw(True)
            
            # if DummyShape is active this frame...
            if DummyShape.status == STARTED:
                # update params
                pass
            
            # if keyMonkey is starting this frame...
            if keyMonkey.status == NOT_STARTED and t >= 0.01-frameTolerance:
                # keep track of start time/frame for later
                keyMonkey.frameNStart = frameN  # exact frame index
                keyMonkey.tStart = t  # local t and not account for scr refresh
                keyMonkey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyMonkey, 'tStartRefresh')  # time at next scr refresh
                # update status
                keyMonkey.status = STARTED
                if PILOTING:
                    # if piloting, keyMonkey will press its key
                    keyMonkey.response = keyMonkey.comp.device.makeResponse(
                        code='left',
                        tDown=t,
                    )
            
            # if keyMonkey is stopping this frame...
            if keyMonkey.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > keyMonkey.tStartRefresh + 0.02-frameTolerance:
                    # keep track of stop time/frame for later
                    keyMonkey.tStop = t  # not accounting for scr refresh
                    keyMonkey.tStopRefresh = tThisFlipGlobal  # on global time
                    keyMonkey.frameNStop = frameN  # exact frame index
                    # update status
                    keyMonkey.status = FINISHED
                    if PILOTING:
                        # if piloting, keyMonkey will release its key
                        keyMonkey.response.duration = t - keyMonkey.response.tDown
            
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
        # check responses
        if SubjectResponse.keys in ['', [], None]:  # No response was made
            SubjectResponse.keys = None
        trials.addData('SubjectResponse.keys',SubjectResponse.keys)
        if SubjectResponse.keys != None:  # we had a response
            trials.addData('SubjectResponse.rt', SubjectResponse.rt)
            trials.addData('SubjectResponse.duration', SubjectResponse.duration)
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Wait" ---
        # create an object to store info about Routine Wait
        Wait = data.Routine(
            name='Wait',
            components=[AnyButton, PressAnyButtonText, keyMonkey_2],
        )
        Wait.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for AnyButton
        AnyButton.keys = []
        AnyButton.rt = []
        _AnyButton_allKeys = []
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
            if AnyButton.status == NOT_STARTED and tThisFlip >= 0.-frameTolerance:
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
            
            # *PressAnyButtonText* updates
            
            # if PressAnyButtonText is starting this frame...
            if PressAnyButtonText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                PressAnyButtonText.frameNStart = frameN  # exact frame index
                PressAnyButtonText.tStart = t  # local t and not account for scr refresh
                PressAnyButtonText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PressAnyButtonText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PressAnyButtonText.started')
                # update status
                PressAnyButtonText.status = STARTED
                PressAnyButtonText.setAutoDraw(True)
            
            # if PressAnyButtonText is active this frame...
            if PressAnyButtonText.status == STARTED:
                # update params
                pass
            
            # if keyMonkey_2 is starting this frame...
            if keyMonkey_2.status == NOT_STARTED and t >= 0.01-frameTolerance:
                # keep track of start time/frame for later
                keyMonkey_2.frameNStart = frameN  # exact frame index
                keyMonkey_2.tStart = t  # local t and not account for scr refresh
                keyMonkey_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyMonkey_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                keyMonkey_2.status = STARTED
                if PILOTING:
                    # if piloting, keyMonkey_2 will press its key
                    keyMonkey_2.response = keyMonkey_2.comp.device.makeResponse(
                        code='space',
                        tDown=t,
                    )
            
            # if keyMonkey_2 is stopping this frame...
            if keyMonkey_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > keyMonkey_2.tStartRefresh + 0.02-frameTolerance:
                    # keep track of stop time/frame for later
                    keyMonkey_2.tStop = t  # not accounting for scr refresh
                    keyMonkey_2.tStopRefresh = tThisFlipGlobal  # on global time
                    keyMonkey_2.frameNStop = frameN  # exact frame index
                    # update status
                    keyMonkey_2.status = FINISHED
                    if PILOTING:
                        # if piloting, keyMonkey_2 will release its key
                        keyMonkey_2.response.duration = t - keyMonkey_2.response.tDown
            
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
        trials.addData('AnyButton.keys',AnyButton.keys)
        if AnyButton.keys != None:  # we had a response
            trials.addData('AnyButton.rt', AnyButton.rt)
            trials.addData('AnyButton.duration', AnyButton.duration)
        # the Routine "Wait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed nReps repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Break_3" ---
    # create an object to store info about Routine Break_3
    Break_3 = data.Routine(
        name='Break_3',
        components=[TakeABreakText, WaitForEsc],
    )
    Break_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for WaitForEsc
    WaitForEsc.keys = []
    WaitForEsc.rt = []
    _WaitForEsc_allKeys = []
    # store start times for Break_3
    Break_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Break_3.tStart = globalClock.getTime(format='float')
    Break_3.status = STARTED
    thisExp.addData('Break_3.started', Break_3.tStart)
    Break_3.maxDuration = None
    # keep track of which components have finished
    Break_3Components = Break_3.components
    for thisComponent in Break_3.components:
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
    
    # --- Run Routine "Break_3" ---
    Break_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *TakeABreakText* updates
        
        # if TakeABreakText is starting this frame...
        if TakeABreakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
        
        # *WaitForEsc* updates
        waitOnFlip = False
        
        # if WaitForEsc is starting this frame...
        if WaitForEsc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            WaitForEsc.frameNStart = frameN  # exact frame index
            WaitForEsc.tStart = t  # local t and not account for scr refresh
            WaitForEsc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(WaitForEsc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'WaitForEsc.started')
            # update status
            WaitForEsc.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(WaitForEsc.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(WaitForEsc.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if WaitForEsc.status == STARTED and not waitOnFlip:
            theseKeys = WaitForEsc.getKeys(keyList=['esc'], ignoreKeys=["escape"], waitRelease=False)
            _WaitForEsc_allKeys.extend(theseKeys)
            if len(_WaitForEsc_allKeys):
                WaitForEsc.keys = _WaitForEsc_allKeys[-1].name  # just the last key pressed
                WaitForEsc.rt = _WaitForEsc_allKeys[-1].rt
                WaitForEsc.duration = _WaitForEsc_allKeys[-1].duration
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
                currentRoutine=Break_3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Break_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Break_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Break_3" ---
    for thisComponent in Break_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Break_3
    Break_3.tStop = globalClock.getTime(format='float')
    Break_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Break_3.stopped', Break_3.tStop)
    # check responses
    if WaitForEsc.keys in ['', [], None]:  # No response was made
        WaitForEsc.keys = None
    thisExp.addData('WaitForEsc.keys',WaitForEsc.keys)
    if WaitForEsc.keys != None:  # we had a response
        thisExp.addData('WaitForEsc.rt', WaitForEsc.rt)
        thisExp.addData('WaitForEsc.duration', WaitForEsc.duration)
    thisExp.nextEntry()
    # the Routine "Break_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
