import React from 'react';

export default function Section({ template, data }) {
    const getAmount = (field, skip) => {
        if (typeof field === 'function') {
            let result = parseInt(field(data));
            return result;
        }
        if (!skip) return field;
        return '';
    };

    return (
        <div className='w-full flex flex-col justify-start items-center gap-0'>
            {template.map((field, index) => (
                <React.Fragment key={field.name}>
                    <div
                        className={`w-full flex flex-row justify-start items-center ${
                            field.indent ? 'pl-4' : ''
                        } gap-2`}
                    >
                        <div className='flex flex-col items-start'>
                            <p className={!field.indent ? 'font-semibold' : ''}>{field.name}</p>
                        </div>
                        <div className='flex flex-col justify-end items-end min-h-full'>
                            {!field.skip && (
                                <p className=''>
                                    {getAmount(field.amount, field.skip)}
                                    {field.unit}
                                </p>
                            )}
                        </div>
                        <div className='flex flex-col justify-end items-end min-h-full grow'>
                            <p className='text-sm font-medium'>
                                {typeof getAmount(field.dailyValue, field.skip) === 'number' &&
                                    getAmount(field.dailyValue, field.skip) + '%'}
                            </p>
                        </div>
                    </div>

                    {index !== template.length - 1 && <hr className='border-black border-[0.5] w-full' />}
                </React.Fragment>
            ))}
        </div>
    );
}
