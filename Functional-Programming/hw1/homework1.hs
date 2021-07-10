module Main where
import System.Environment(getArgs)
import Data.Char(intToDigit, digitToInt, chr, ord)

-- Function to parse input arguments
-- Input list of strings (command line args), output string to be displayed
parseArgs :: [String] -> [Char]
parseArgs args
    | head args == "d2c" = digitsToChars (read $ args !! 1) (read $ args !! 2)
    | head args == "c2d" = charsToDigits (read $ args !! 1) (args !! 2)
    | head args == "n2l" = getNumberToList (read $ args !! 1) (read $ args !! 2)
    | head args == "l2n" = getListToNumber (read $ args !! 1) (map read $ (tail . tail) args)
    | head args == "add" = addDecimals (read $ args !! 1) (read $ args !! 2) (read $ args !! 3)
    | otherwise = "Unrecognized operation!"

-- Function to convert digits to characters
-- Input base and decimal value, output string to be displayed
digitsToChars :: Int -> Int -> [Char]
digitsToChars base dVal
    | base <= dVal = error ": invalid digit"    -- Throw error if base is smaller than the value
    | base > 36 = error ": invalid digit"   -- Throw error if base is larger than 36
    | otherwise = show $ mapDecToHex dVal   -- Convert digit

-- A map to convert digits to characters
-- Input decimal value, output character representation of the value (max base 36)
mapDecToHex :: Int -> Char
mapDecToHex x
    | x < 10 = intToDigit x
    | x < 36 = chr (55 + x)
    | otherwise = error ": invalid digit" -- Throw error if base is larger than 36

-- Function to convert characters to digits
-- Input base and number string in the given base, output string to be displayed
charsToDigits :: Int -> [Char] -> [Char]
charsToDigits base cVal
    | base <= read (mapHexToDec cVal) = error ": invalid digit" -- Throw error if base is smaller than the value
    | base > 36 = error ": invalid digit" -- Throw error if base is larger than 36
    | otherwise = mapHexToDec cVal  -- Convert digit

-- A map to convert characters to digits
-- Input decimal value, output character representation of the value (max base 36)
mapHexToDec :: [Char] -> [Char]
mapHexToDec x
    | ordX < 65 = x
    | otherwise = show (ordX - 55)
        where
            ordX = ord $ head x

-- A convenience function to convert number list to string
getNumberToList :: Int -> Int -> [Char]
getNumberToList base num = show $ reverse $ numberToList base num

-- Converts a number to a given base
-- Input base to be converted and number to be converted in decimal, output list of integers 
--   that represent the input number in given base
numberToList :: Int -> Int -> [Int]
numberToList base num
    | base <= 0 || num <= 0 = []    -- Base case
    | otherwise = mod num base:numberToList base (div num base) -- Convert decimal to base

-- A convenience function to convert list of numbers to string
getListToNumber :: Int -> [Int] -> [Char]
getListToNumber base nums = show $ listToNumber base nums

-- Converts a list of numbers in a given base to decimal number
-- Input base to be converted from and number list to be converted, output integer in decimal
listToNumber :: Int -> [Int] -> Int
listToNumber base nums
    | base <= 0 = 0 -- Base case
    | otherwise = baseConverter base nums (length nums - 1)
        where
            baseConverter :: Int -> [Int] -> Int -> Int
            baseConverter _ _ (-1) = 0  -- Base case
            baseConverter base (x:xs) pow = x*base^pow + baseConverter base xs (pow - 1)    -- Convert from base to decimal

-- Calls previously defined functions in the required order and appends them in a printable format
-- Input base to be converted, first decimal value, second decimal value, output string to be displayed
addDecimals :: Int -> Int -> Int -> [Char]
addDecimals base num1 num2 =
    getNumberToList base num1 ++ "\n\"" ++ reverse (map mapDecToHex (numberToList base num1)) ++ "\"\n" ++
    getNumberToList base num2 ++ "\n\"" ++ reverse (map mapDecToHex (numberToList base num2)) ++ "\"\n" ++
    getNumberToList base (num1 + num2) ++ "\n" ++ show (num1 + num2)

-- The main loop that takes in command line arguments and prints the output
main :: IO ()
main = do
    args <- getArgs
    putStrLn $ parseArgs args