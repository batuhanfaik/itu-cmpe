module Main where
import System.Environment ( getArgs )
import Data.Char(intToDigit, digitToInt)

checkArgs [a, b, c] = parseArgs [a, b, c]
checkArgs _ = putStrLn "At least three input arguments required"

parseArgs [a, b, c]
    | a == "d2c" = putStrLn "Selam"
    | otherwise = putStrLn "Other"

main = do
    args <- getArgs
    checkArgs args