module Main where
import System.Environment ( getArgs )
import Data.Char(intToDigit, digitToInt)

printArgs [a, b, c] = putStrLn (a ++ " " ++ b ++ " " ++ c)
printArgs _ = putStrLn "At least three input arguments required"

main = do
    args <- getArgs
    printArgs args