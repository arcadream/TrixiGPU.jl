# Create some macros to assist the testing process

# Macro to test the approximate equality of arrays from GPU and CPU with NaNs
macro test_approx(expr)
    # Parse the expression and check that it is of the form 
    # @test_approx (array1, array2)
    if expr.head != :tuple || length(expr.args) != 2
        error("Incorrect usage. Expected syntax: @test_approx(array1, array2)")
    end

    local array1 = esc(expr.args[1])
    local array2 = esc(expr.args[2])

    quote
        # Convert to arrays to avoid using CUDA.@allowscalar 
        # to access the elements of some arrays
        local _array1 = Array($array1)
        local _array2 = Array($array2)

        # Check if the arrays have NaN
        local has_nan_arr1 = any(isnan, _array1)
        local has_nan_arr2 = any(isnan, _array2)

        if has_nan_arr1 && has_nan_arr2 # both have NaN
            # Condition 1: Check if NaNs are at the same position
            local cond1 = isnan.(_array1) == isnan.(_array2)

            # Replace NaNs with 0.0
            local __array1 = replace(_array1, NaN => 0.0)
            local __array2 = replace(_array2, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = __array1 ≈ __array2

            @test cond1 && cond2
        elseif !has_nan_arr1 && !has_nan_arr2 # neither has NaN

            # Direct comparison
            @test _array1 ≈ _array2

            # Truth table for below cases
            # -------------------------------
            #   Entry   |   Entry   | Result
            # -------------------------------
            #    NaN    |   zero    |   1
            #    NaN    |  non-zero |   0
            #  non-NaN  |   zero    |   1
            #  non-NaN  |  non-zero |   1
            # -------------------------------
        elseif has_nan_arr1 # only the first array has NaN
            # Condition 1: Check truth table above
            local cond1 = all(.!(isnan.(_array1) .&& (_array2 .!= 0.0)))

            # Replace NaNs with 0.0
            local __array1 = replace(_array1, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = _array2 ≈ __array1

            @test cond1 && cond2
        elseif has_nan_arr2 # only the second array has NaN
            # Condition 1: Check truth table above
            local cond1 = all(.!(isnan.(_array2) .&& (_array1 .!= 0.0)))

            # Replace NaNs with 0.0
            local __array2 = replace(_array2, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = _array1 ≈ __array2

            @test cond1 && cond2
        end
    end
end

# Macro to test the exact equality of two arrays, which can be from the CPU, GPU, 
# or a combination of both
macro test_equal(expr)
    # Parse the expression and check that it is of the form 
    # @test_equal (array1, array2)
    if expr.head != :tuple || length(expr.args) != 2
        error("Incorrect usage. Expected syntax: @test_approx(array1, array2)")
    end

    local array1 = esc(expr.args[1])
    local array2 = esc(expr.args[2])

    quote
        # Convert to arrays to avoid using CUDA.@allowscalar 
        # to access the elements of some arrays
        local _array1 = Array($array1)
        local _array2 = Array($array2)

        @test _array1 == _array2
    end
end
