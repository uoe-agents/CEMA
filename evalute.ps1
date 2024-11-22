# Load the 'xavi' miniconda environment
& "C:\Users\Balint\miniconda3\Scripts\activate" xavi

# Define the list of scenarios as integers from 1 to 4 and from 9 to 11
$scenarios = 1..4 + 9..11

# Define a dictionary to map scenarios to query numbers
$scenarioQueryMap = @{
    1 = 0..3
    2 = @(0, 1)
    3 = @(1)
    4 = @(0)
    9 = @(0, 1)
    10 = @(0, 1)
    11 = @(0)
}

Write-Host "Starting evaluation of scenarios..."

if (-not (Test-Path "output")) {
    Write-Host "The 'output' folder does not exist. Please use scenarios/run.py to generate the outputs for evaluation first."
    exit
}

# Loop through each scenario and query number
foreach ($scenario in $scenarios) {
    $queryNumbers = $scenarioQueryMap[$scenario]
    foreach ($queryNumber in $queryNumbers) {
        # Log the current scenario and query number
        Write-Host "Evaluating scenario $scenario with query $queryNumber ..."
        
        # Call the evaluate.py script with the scenario and query number
        python scenarios/evaluation.py --sid $scenario --qid $queryNumber

        # Add empty line for better readability
        Write-Host ""
    }
}

Write-Host "Evaluation complete."
